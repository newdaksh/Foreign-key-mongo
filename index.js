// index.js
// Single-file server that:
// - connects to MongoDB
// - exposes /search/users, /search/events, /search/dating (plain-English "q" parsing)
// - optional populate=true to include user docs in events/dating
// - optional /openai/function to accept OpenAI function_call json and run the corresponding search
// - attempts to register MCP tools if @modelcontextprotocol/sdk is installed
//
// Run instructions:
// 1) npm init -y
// 2) npm i express mongodb dotenv
//    optional: npm i @modelcontextprotocol/sdk zod openai
// 3) add "type": "module" to package.json (or run node with ESM enabled)
// 4) create .env with:
//    MONGO_URI="mongodb://localhost:27017"
//    DB_NAME="mydb"
//    PORT=8000
//    API_KEY="change_this_to_secret"
//    ALLOW_PII=false   # set to "true" only for dev/testing to allow email/salary in responses
//    OPENAI_API_KEY=... (optional)
// 5) node index.js
//
// Postman examples:
// POST http://localhost:8000/search/users
// Headers: Authorization: Bearer <API_KEY>, Content-Type: application/json
// Body: {"q":"female UX Designers in Bengaluru older than 25", "limit": 10}
//
// GET /search/events?populate=true  (body: {"q":"Tech Meetup Bengaluru", "limit":5})

import express from "express";
import OpenAI from "openai";
import dotenv from "dotenv";
import { MongoClient, ObjectId } from "mongodb";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
dotenv.config();

const app = express();
app.use(express.json());

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017";
const DB_NAME = process.env.DB_NAME || "mydb";
const PORT = Number(process.env.PORT || 8000);
const API_KEY = process.env.API_KEY || "testkey123";
const ALLOW_PII = process.env.ALLOW_PII === "true";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

// ----- simple API key middleware -----
app.use((req, res, next) => {
  // allow health check
  if (req.path === "/health") return next();
  const auth = req.header("Authorization") || "";
  const token = auth.replace(/^Bearer\s+/i, "");
  if (!token || token !== API_KEY) {
    return res
      .status(401)
      .json({ error: "Unauthorized. Set Authorization: Bearer <API_KEY>" });
  }
  next();
});

// ----- utilities -----
function hideSensitive(doc) {
  if (!ALLOW_PII) {
    const { Salary, email, ...rest } = doc;
    return rest;
  }
  return doc;
}

// Build a Mongo filter from a simple natural-language query string.
// This is intentionally conservative and safe. Expand patterns as needed.
function buildFilterFromQueryForUsers(q) {
  return parsePromptToMongoQuery(q, "users");
}

// ----- Date filter utility function -----
function buildDateFilter(q, dateField) {
  const query = q.toLowerCase();

  // Check for date patterns
  let dateFilter = null;

  // Pattern: "on DD-MM-YYYY" or "DD-MM-YYYY"
  const exactDateMatch = query.match(/(?:on\s+)?(\d{1,2})-(\d{1,2})-(\d{4})/);
  if (exactDateMatch) {
    const [, day, month, year] = exactDateMatch;
    const targetDate = `${year}-${month.padStart(2, "0")}-${day.padStart(
      2,
      "0"
    )}`;
    dateFilter = {};
    dateFilter[dateField] = { $regex: targetDate, $options: "i" };
  }

  // Pattern: "in MM-YYYY" or "in YYYY"
  const monthYearMatch = query.match(/(?:in\s+)?(\d{1,2})-(\d{4})/);
  if (monthYearMatch && !exactDateMatch) {
    const [, month, year] = monthYearMatch;
    const targetPattern = `${year}-${month.padStart(2, "0")}`;
    dateFilter = {};
    dateFilter[dateField] = { $regex: targetPattern, $options: "i" };
  }

  // Pattern: "in YYYY"
  const yearMatch = query.match(/(?:in\s+)?(\d{4})/);
  if (yearMatch && !exactDateMatch && !monthYearMatch) {
    const [, year] = yearMatch;
    dateFilter = {};
    dateFilter[dateField] = { $regex: year, $options: "i" };
  }

  // Pattern: "today", "yesterday", "tomorrow"
  const today = new Date();
  if (query.includes("today")) {
    const todayStr = today.toISOString().split("T")[0];
    dateFilter = {};
    dateFilter[dateField] = { $regex: todayStr, $options: "i" };
  } else if (query.includes("yesterday")) {
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split("T")[0];
    dateFilter = {};
    dateFilter[dateField] = { $regex: yesterdayStr, $options: "i" };
  } else if (query.includes("tomorrow")) {
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    const tomorrowStr = tomorrow.toISOString().split("T")[0];
    dateFilter = {};
    dateFilter[dateField] = { $regex: tomorrowStr, $options: "i" };
  }

  // Pattern: "this week", "next week", "last week"
  if (query.includes("this week")) {
    const startOfWeek = new Date(today);
    startOfWeek.setDate(today.getDate() - today.getDay());
    const endOfWeek = new Date(startOfWeek);
    endOfWeek.setDate(startOfWeek.getDate() + 6);

    dateFilter = {};
    dateFilter[dateField] = {
      $gte: startOfWeek.toISOString().split("T")[0],
      $lte: endOfWeek.toISOString().split("T")[0],
    };
  }

  return dateFilter;
}

function buildFilterFromQueryForEvents(q) {
  return parsePromptToMongoQuery(q, "events");
}

function buildFilterFromQueryForDating(q) {
  return parsePromptToMongoQuery(q, "dating");
}
// ----- Dynamic prompt-to-query parser -----
function parsePromptToMongoQuery(prompt, type) {
  if (!openai || !prompt) return {};
  const systemMsg = `You are an assistant that converts natural language search prompts into MongoDB query objects for the ${type} collection. Only return a valid MongoDB query object, no explanation.`;
  const userMsg = `Prompt: ${prompt}`;
  return (async () => {
    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
          { role: "system", content: systemMsg },
          { role: "user", content: userMsg },
        ],
        temperature: 0.2,
        max_tokens: 300,
      });
      const text = completion.choices[0].message.content.trim();
      const jsonStart = text.indexOf("{");
      const jsonEnd = text.lastIndexOf("}") + 1;
      if (jsonStart !== -1 && jsonEnd !== -1) {
        return JSON.parse(text.substring(jsonStart, jsonEnd));
      }
      return {};
    } catch (err) {
      console.error("NLP parse error:", err);
      return {};
    }
  })();
}

// ----- Clean search functions for smart endpoint -----
async function searchUsersClean(q) {
  const filter = await buildFilterFromQueryForUsers(q);
  const projection = ALLOW_PII ? {} : { Salary: 0, email: 0 };
  const results = await db
    .collection("users")
    .find(filter, { projection })
    .limit(10)
    .toArray();
  return results.map((user) => ({
    name: user.Name,
    age: user.age,
    gender: user.Gender,
    location: user.Location,
    occupation: user.Occupation,
    ...(ALLOW_PII && { salary: user.Salary, email: user.email }),
  }));
}

async function searchEventsClean(q) {
  console.log("searchEventsClean called with query:", q);
  const baseFilter = await buildFilterFromQueryForEvents(q);
  console.log("Generated filter:", JSON.stringify(baseFilter, null, 2));

  // Resolve user name matches and add to OR if present
  const userMatches = await db
    .collection("users")
    .find({ Name: { $regex: q, $options: "i" } }, { projection: { _id: 1 } })
    .toArray();
  if (userMatches.length) {
    baseFilter.$or.push({
      participant_ids: { $in: userMatches.map((u) => u._id) },
    });
  }

  const results = await db
    .collection("events")
    .find(baseFilter)
    .limit(10)
    .toArray();

  console.log("Raw results from database:", results.length, "events found");

  // Populate participant details
  for (let event of results) {
    if (event.participant_ids && Array.isArray(event.participant_ids)) {
      const participants = await db
        .collection("users")
        .find(
          { _id: { $in: event.participant_ids } },
          {
            projection: {
              Name: 1,
              Location: 1,
              Occupation: 1,
              age: 1,
              Gender: 1,
            },
          }
        )
        .toArray();
      event.participants = participants.map((p) => ({
        name: p.Name,
        location: p.Location,
        occupation: p.Occupation,
        age: p.age,
        gender: p.Gender,
      }));
    }
  }

  return results.map((event) => ({
    eventType: event.Event_type,
    location: event.Event_location,
    eventDate: event.Event_date,
    participants: event.participants || [],
  }));
}

async function searchDatingClean(q) {
  console.log("searchDatingClean called with query:", q);
  const baseFilter = await buildFilterFromQueryForDating(q);
  console.log("Generated dating filter:", JSON.stringify(baseFilter, null, 2));

  // Resolve user name matches
  const userMatches = await db
    .collection("users")
    .find({ Name: { $regex: q, $options: "i" } }, { projection: { _id: 1 } })
    .toArray();
  if (userMatches.length) {
    baseFilter.$or.push({
      Male_id: { $in: userMatches.map((u) => u._id) },
    });
    baseFilter.$or.push({
      Female_id: { $in: userMatches.map((u) => u._id) },
    });
  }

  // Try both "datings" and "Dating" collection names
  let results = [];
  try {
    results = await db
      .collection("datings")
      .find(baseFilter)
      .limit(10)
      .toArray();
    console.log("Results from 'datings' collection:", results.length);
  } catch (err) {
    console.log("'datings' collection not found, trying 'Dating'");
    try {
      results = await db
        .collection("Dating")
        .find(baseFilter)
        .limit(10)
        .toArray();
      console.log("Results from 'Dating' collection:", results.length);
    } catch (err2) {
      console.log("Neither 'datings' nor 'Dating' collection found");
      return [];
    }
  }

  // Populate user details
  for (let profile of results) {
    if (profile.Male_id) {
      const male = await db.collection("users").findOne(
        { _id: profile.Male_id },
        {
          projection: {
            Name: 1,
            age: 1,
            Location: 1,
            Occupation: 1,
            Gender: 1,
          },
        }
      );
      profile.male = male
        ? {
            name: male.Name,
            age: male.age,
            location: male.Location,
            occupation: male.Occupation,
            gender: male.Gender,
          }
        : null;
    }
    if (profile.Female_id) {
      const female = await db.collection("users").findOne(
        { _id: profile.Female_id },
        {
          projection: {
            Name: 1,
            age: 1,
            Location: 1,
            Occupation: 1,
            Gender: 1,
          },
        }
      );
      profile.female = female
        ? {
            name: female.Name,
            age: female.age,
            location: female.Location,
            occupation: female.Occupation,
            gender: female.Gender,
          }
        : null;
    }
  }

  return results.map((dating) => ({
    location: dating.Dating_location,
    datingDate: dating.Dating_Date,
    male: dating.male,
    female: dating.female,
  }));
}

// ----- MongoDB connection & server start -----
let db;
let client;

async function startServer() {
  client = new MongoClient(MONGO_URI, { useUnifiedTopology: true });
  await client.connect();
  db = client.db(DB_NAME);
  console.log("Connected to MongoDB:", MONGO_URI, " DB:", DB_NAME);

  // Initialize MCP Server
  setupMCPServer();

  // ---- Express endpoints ----

  // Health
  app.get("/health", (req, res) => res.json({ ok: true }));

  // SEARCH USERS: body { q: string, limit?: number }
  app.post("/search/users", async (req, res) => {
    try {
      const { q = "", limit = 10 } = req.body || {};
      const l = Math.min(Number(limit) || 10, 100);
      const filter = await buildFilterFromQueryForUsers(q);
      const projection = ALLOW_PII ? {} : { Salary: 0, email: 0 };
      const results = await db
        .collection("users")
        .find(filter, { projection })
        .limit(l)
        .toArray();
      const safe = results.map(hideSensitive);
      return res.json({ count: safe.length, results: safe });
    } catch (err) {
      console.error("search/users error", err);
      return res.status(500).json({ error: "Server error" });
    }
  });

  // SEARCH EVENTS: body { q: string, limit?: number }, query param populate=true to populate participant docs
  app.post("/search/events", async (req, res) => {
    try {
      const { q = "", limit = 10 } = req.body || {};
      const populate = req.query.populate === "true";
      const l = Math.min(Number(limit) || 10, 100);
      const baseFilter = await buildFilterFromQueryForEvents(q);

      // Resolve user name matches and add to OR if present
      const userMatches = await db
        .collection("users")
        .find(
          { Name: { $regex: q, $options: "i" } },
          { projection: { _id: 1 } }
        )
        .toArray();
      if (userMatches.length) {
        baseFilter.$or.push({
          participant_ids: { $in: userMatches.map((u) => u._id) },
        });
      }
      const cursor = db.collection("events").find(baseFilter).limit(l);
      let docs = await cursor.toArray();

      if (populate && docs.length) {
        // populate participants using $lookup per event (manual populate)
        const userIds = Array.from(
          new Set(
            docs.flatMap((d) =>
              (d.participant_ids || []).map((id) => id.toString())
            )
          )
        ).map((s) => new ObjectId(s));
        const users = await db
          .collection("users")
          .find(
            { _id: { $in: userIds } },
            { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
          )
          .toArray();
        const usersById = new Map(
          users.map((u) => [u._id.toString(), hideSensitive(u)])
        );
        docs = docs.map((ev) => {
          const participants = (ev.participant_ids || []).map(
            (id) => usersById.get(id.toString()) || { _id: id }
          );
          return { ...ev, participants };
        });
      } else {
        // hide sensitive fields in participant_ids responses? nothing to do
      }

      return res.json({ count: docs.length, results: docs });
    } catch (err) {
      console.error("search/events error", err);
      return res.status(500).json({ error: "Server error" });
    }
  });

  // SEARCH DATING: body { q: string, limit?: number }, query param populate=true to populate Male/Female user docs
  app.post("/search/dating", async (req, res) => {
    try {
      const { q = "", limit = 10 } = req.body || {};
      const populate = req.query.populate === "true";
      const l = Math.min(Number(limit) || 10, 100);
      const baseFilter = await buildFilterFromQueryForDating(q);

      const userMatches = await db
        .collection("users")
        .find(
          { Name: { $regex: q, $options: "i" } },
          { projection: { _id: 1 } }
        )
        .toArray();
      if (userMatches.length) {
        baseFilter.$or.push({
          Male_id: { $in: userMatches.map((u) => u._id) },
        });
        baseFilter.$or.push({
          Female_id: { $in: userMatches.map((u) => u._id) },
        });
      }

      let docs = await db
        .collection("Dating")
        .find(baseFilter)
        .limit(l)
        .toArray();

      if (populate && docs.length) {
        const userIds = Array.from(
          new Set(
            docs.flatMap((d) =>
              [d.Male_id, d.Female_id]
                .filter(Boolean)
                .map((id) => id.toString())
            )
          )
        ).map((s) => new ObjectId(s));
        const users = await db
          .collection("users")
          .find(
            { _id: { $in: userIds } },
            { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
          )
          .toArray();
        const usersById = new Map(
          users.map((u) => [u._id.toString(), hideSensitive(u)])
        );
        docs = docs.map((dt) => ({
          ...dt,
          Male: dt.Male_id ? usersById.get(dt.Male_id.toString()) : null,
          Female: dt.Female_id ? usersById.get(dt.Female_id.toString()) : null,
        }));
      }

      return res.json({ count: docs.length, results: docs });
    } catch (err) {
      console.error("search/dating error", err);
      return res.status(500).json({ error: "Server error" });
    }
  });

  // SMART SEARCH: automatically detects search type and returns clean results
  app.post("/search/debug", async (req, res) => {
    try {
      const query =
        req.body.query?.toLowerCase() || req.body.q?.toLowerCase() || "";

      // Detect what type of search based on keywords
      let searchType = "users"; // default

      if (
        query.includes("event") ||
        query.includes("meetup") ||
        query.includes("conference") ||
        query.includes("gathering")
      ) {
        searchType = "events";
      } else if (
        query.includes("dating") ||
        query.includes("match") ||
        query.includes("relationship") ||
        query.includes("couple")
      ) {
        searchType = "dating";
      }

      let results = [];
      let mongoQuery = {};
      const searchQ = req.body.query || req.body.q || "";
      if (searchType === "users") {
        mongoQuery = await buildFilterFromQueryForUsers(searchQ);
        results = await searchUsersClean(searchQ);
      } else if (searchType === "events") {
        mongoQuery = await buildFilterFromQueryForEvents(searchQ);
        results = await searchEventsClean(searchQ);
      } else if (searchType === "dating") {
        mongoQuery = await buildFilterFromQueryForDating(searchQ);
        results = await searchDatingClean(searchQ);
      }
      return res.json({
        searchType: searchType,
        query: searchQ,
        mongoQuery,
        count: results.length,
        results: results,
      });
    } catch (err) {
      console.error("smart search error", err);
      return res
        .status(500)
        .json({ error: "Search failed", message: err.message });
    }
  });

  // OPTIONAL: Accept an OpenAI function_call output and execute the search (useful when using function-calling)
  // Body: { function_call: { name: "search_users"|"search_events"|"search_dating", arguments: <JSON string> } }
  app.post("/openai/function", async (req, res) => {
    try {
      if (!OPENAI_API_KEY)
        return res
          .status(400)
          .json({ error: "OPENAI_API_KEY not configured on server." });
      const { function_call } = req.body || {};
      if (!function_call || !function_call.name)
        return res.status(400).json({ error: "invalid function_call" });
      const args = function_call.arguments
        ? JSON.parse(function_call.arguments)
        : {};
      // map to our endpoints
      if (function_call.name === "search_users") {
        req.body = { q: args.q || args.query || "", limit: args.limit || 10 };
        return app._router.handle(req, res, () => {});
      }
      if (function_call.name === "search_events") {
        req.body = { q: args.q || args.query || "", limit: args.limit || 10 };
        return app._router.handle(req, res, () => {});
      }
      if (function_call.name === "search_dating") {
        req.body = { q: args.q || args.query || "", limit: args.limit || 10 };
        return app._router.handle(req, res, () => {});
      }
      return res.status(400).json({ error: "unknown function name" });
    } catch (err) {
      console.error("openai/function error", err);
      return res.status(500).json({ error: "Server error" });
    }
  });

  // Start Express
  app.listen(PORT, () => console.log(`API listening http://localhost:${PORT}`));

  // ----- Attempt to register MCP tools automatically (optional) -----
  // If @modelcontextprotocol/sdk is NOT installed, this block won't crash the server.
  (async () => {
    try {
      const { McpServer } = await import(
        "@modelcontextprotocol/sdk/server/mcp.js"
      );
      const types = await import("@modelcontextprotocol/sdk/types.js").catch(
        () => null
      );
      const { z } = await import("zod").catch(() => ({ z: null }));
      if (!McpServer) {
        console.log("MCP SDK found but McpServer import failed.");
        return;
      }
      const mcp = new McpServer({ name: "mongo-mcp", version: "0.1.0" });
      console.log("MCP SDK found — registering tools...");

      // register search_users
      mcp.registerTool(
        {
          name: "search_users",
          title: "Search Users",
          description:
            "Search users by plain text q (name/location/occupation) and return limited docs.",
          inputSchema: z.object({
            q: z.string().min(0),
            limit: z.number().int().min(1).max(100).optional(),
          }),
        },
        async ({ q, limit = 10 }) => {
          const filter = await buildFilterFromQueryForUsers(q);
          const docs = await db
            .collection("users")
            .find(filter, {
              projection: ALLOW_PII ? {} : { Salary: 0, email: 0 },
            })
            .limit(limit)
            .toArray();
          const safe = docs.map(hideSensitive);
          // types.CallToolResult may or may not exist depending on SDK version; fallback to a plain object
          if (types && types.CallToolResult) {
            return types.CallToolResult({
              content: [
                types.TextContent({
                  type: "text",
                  text: `Found ${safe.length} results.`,
                }),
                types.StructuredContent({
                  type: "application/json",
                  data: safe,
                }),
              ],
            });
          }
          return { text: `Found ${safe.length} results.`, data: safe };
        }
      );

      // register search_events
      mcp.registerTool(
        {
          name: "search_events",
          title: "Search Events",
          description:
            "Search events by q; resolves name matches to participant_ids automatically.",
          inputSchema: z.object({
            q: z.string().min(0),
            limit: z.number().int().min(1).max(100).optional(),
          }),
        },
        async ({ q, limit = 10 }) => {
          const baseFilter = await buildFilterFromQueryForEvents(q);
          const userMatches = await db
            .collection("users")
            .find(
              { Name: { $regex: q, $options: "i" } },
              { projection: { _id: 1 } }
            )
            .toArray();
          if (userMatches.length)
            baseFilter.$or.push({
              participant_ids: { $in: userMatches.map((u) => u._id) },
            });
          const docs = await db
            .collection("events")
            .find(baseFilter)
            .limit(limit)
            .toArray();
          return { text: `Found ${docs.length} events.`, data: docs };
        }
      );

      // register search_dating
      mcp.registerTool(
        {
          name: "search_dating",
          title: "Search Dating Records",
          description:
            "Search dating records by q; resolves name matches to Male_id/Female_id automatically.",
          inputSchema: z.object({
            q: z.string().min(0),
            limit: z.number().int().min(1).max(100).optional(),
          }),
        },
        async ({ q, limit = 10 }) => {
          const baseFilter = await buildFilterFromQueryForDating(q);
          const userMatches = await db
            .collection("users")
            .find(
              { Name: { $regex: q, $options: "i" } },
              { projection: { _id: 1 } }
            )
            .toArray();
          if (userMatches.length) {
            baseFilter.$or.push({
              Male_id: { $in: userMatches.map((u) => u._id) },
            });
            baseFilter.$or.push({
              Female_id: { $in: userMatches.map((u) => u._id) },
            });
          }
          const docs = await db
            .collection("Dating")
            .find(baseFilter)
            .limit(limit)
            .toArray();
          return { text: `Found ${docs.length} dating records.`, data: docs };
        }
      );

      // Start MCP server transport (HTTP SSE default) — environment variable MCP_PORT optional
      const MCP_PORT = Number(process.env.MCP_PORT || 9000);
      await mcp.start({ host: "0.0.0.0", port: MCP_PORT });
      console.log("MCP server running on port", MCP_PORT);
    } catch (err) {
      console.log(
        "MCP tools not registered (optional). To enable MCP: npm i @modelcontextprotocol/sdk zod"
      );
      // console.debug(err);
    }
  })();
}

// ---- MCP Server Setup ----
function setupMCPServer() {
  const server = new Server(
    {
      name: "mongo-search-server",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Register MongoDB search tools
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: [
        {
          name: "search_users",
          description:
            "Search users with plain English queries (e.g., 'female UX designers in Bengaluru')",
          inputSchema: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "Plain English search query for users",
              },
              limit: {
                type: "number",
                description:
                  "Maximum number of results (default: 10, max: 100)",
                default: 10,
              },
            },
            required: ["query"],
          },
        },
        {
          name: "search_events",
          description: "Search events with plain English queries",
          inputSchema: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "Plain English search query for events",
              },
              limit: {
                type: "number",
                description:
                  "Maximum number of results (default: 10, max: 100)",
                default: 10,
              },
              populate: {
                type: "boolean",
                description: "Include full user details for participants",
                default: false,
              },
            },
            required: ["query"],
          },
        },
        {
          name: "search_dating",
          description: "Search dating profiles with plain English queries",
          inputSchema: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "Plain English search query for dating profiles",
              },
              limit: {
                type: "number",
                description:
                  "Maximum number of results (default: 10, max: 100)",
                default: 10,
              },
              populate: {
                type: "boolean",
                description: "Include full user details",
                default: false,
              },
            },
            required: ["query"],
          },
        },
      ],
    };
  });

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      switch (name) {
        case "search_users": {
          const { query = "", limit = 10 } = args;
          const l = Math.min(Number(limit) || 10, 100);
          const filter = buildFilterFromQueryForUsers(query);
          const projection = ALLOW_PII ? {} : { Salary: 0, email: 0 };
          const results = await db
            .collection("users")
            .find(filter, { projection })
            .limit(l)
            .toArray();

          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(
                  { results, count: results.length, query: filter },
                  null,
                  2
                ),
              },
            ],
          };
        }

        case "search_events": {
          const { query = "", limit = 10, populate = false } = args;
          const l = Math.min(Number(limit) || 10, 100);
          const filter = buildFilterFromQueryForEvents(query);
          let results = await db
            .collection("events")
            .find(filter)
            .limit(l)
            .toArray();

          if (populate) {
            // Populate user details for participants
            for (let event of results) {
              if (event.participants && Array.isArray(event.participants)) {
                const userIds = event.participants.map((p) =>
                  typeof p === "string" ? new ObjectId(p) : p
                );
                const users = await db
                  .collection("users")
                  .find(
                    { _id: { $in: userIds } },
                    { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
                  )
                  .toArray();
                event.participantDetails = users;
              }
            }
          }

          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(
                  {
                    results,
                    count: results.length,
                    query: filter,
                    populated: populate,
                  },
                  null,
                  2
                ),
              },
            ],
          };
        }

        case "search_dating": {
          const { query = "", limit = 10, populate = false } = args;
          const l = Math.min(Number(limit) || 10, 100);
          const filter = buildFilterFromQueryForDating(query);
          let results = await db
            .collection("dating")
            .find(filter)
            .limit(l)
            .toArray();

          if (populate) {
            // Populate user details
            for (let profile of results) {
              if (profile.userId) {
                const userId =
                  typeof profile.userId === "string"
                    ? new ObjectId(profile.userId)
                    : profile.userId;
                const user = await db
                  .collection("users")
                  .findOne(
                    { _id: userId },
                    { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
                  );
                profile.userDetails = user;
              }
            }
          }

          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(
                  {
                    results,
                    count: results.length,
                    query: filter,
                    populated: populate,
                  },
                  null,
                  2
                ),
              },
            ],
          };
        }

        default:
          throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
      }
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Error executing tool ${name}: ${error.message}`
      );
    }
  });

  // Start MCP server if running in MCP mode
  if (process.argv.includes("--mcp")) {
    const transport = new StdioServerTransport();
    server.connect(transport);
    console.log("MCP Server started with stdio transport");
  } else {
    console.log("MCP tools registered (use --mcp flag to start MCP server)");
  }
}

startServer().catch((err) => {
  console.error("Fatal error starting server:", err);
  process.exit(1);
});

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("Shutting down...");
  try {
    if (client) await client.close();
  } catch (e) {}
  process.exit(0);
});
