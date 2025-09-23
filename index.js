// index.js
// Single-file server that:
// - connects to MongoDB
// - exposes /search/users, /search/events, /search/dating (plain-English "q" parsing)
// - optional populate=true to include user docs in events/dating
//
// Run instructions:
// 1) npm init -y
// 2) npm i express mongodb dotenv openai
//    optional: npm i @modelcontextprotocol/sdk zod
// 3) add "type": "module" to package.json (or run node with ESM enabled)
// 4) create .env with:
//    MONGO_URI="mongodb://localhost:27017"
//    DB_NAME="mydb"
//    PORT=8000
//    API_KEY="change_this_to_secret"
//    ALLOW_PII=false   # set to "true" only for dev/testing to allow email/salary in responses
//    OPENAI_API_KEY=...
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

// SYSTEM PROMPT update
const SYSTEM_PROMPT = `
SYSTEM ROLE: You are an assistant whose only job is to convert a user's natural-language search prompt into a single valid JSON object that is a MongoDB query for a specific collection. RETURN ONLY A JSON OBJECT (no explanation, no extra text, no code fences). The output MUST be a syntactically valid JSON object that MongoDB accepts when parsed.


CRITICAL RULES FOR COLLECTION HINTS:
- CollectionHint "events" → Query for "events" collection using "Event_date", "Event_location", "Event_type" fields
- CollectionHint "dating" → Query for "datings" collection using "Dating_Date", "Dating_location", "Male_id", "Female_id" fields  
- CollectionHint "users" → Query for "users" collection using "Name", "Gender", "Location", "DOB", "Salary", "Occupation", "email" fields

MANDATORY DATE HANDLING (applies to ALL collections):
- NEVER use $expr, $month, $year - ALWAYS use $gte/$lte ranges with $dateFromString
- Month queries MUST generate date ranges with desired results:
  * "events in november" → { "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-11-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-11-30T23:59:59Z", "timezone": "UTC" } } } }
  * "datings in december" → { "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-12-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-12-31T23:59:59Z", "timezone": "UTC" } } } }

DATING COLLECTION SPECIAL RULES:
- ANY prompt with "dating", "datings", "date", "dates" + month name → Dating_Date range query
- NEVER return {} for dating collection when month is mentioned
- Follow EXACT same pattern as events collection but use "Dating_Date" field with user desired result

EXAMPLES (MANDATORY PATTERNS):
- CollectionHint "events", Prompt "events in november" → { "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-11-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-11-30T23:59:59Z", "timezone": "UTC" } } } }
- CollectionHint "dating", Prompt "datings in december" → { "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-12-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-12-31T23:59:59Z", "timezone": "UTC" } } } }
- CollectionHint "dating", Prompt "datings in june" → { "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-06-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-06-30T23:59:59Z", "timezone": "UTC" } } } }

OUTPUT RULES:
- Return ONLY valid JSON object
- Use exact field names: "Event_date", "Dating_Date", "DOB"
- Use $dateFromString for ALL dates
- Use $regex with $options:"i" for text searches
- Quote all MongoDB operators with double quotes

DATING COLLECTION RULES (HIGHEST PRIORITY):
- If CollectionHint is "dating", you are generating a query for the "datings" MongoDB collection.
- If the prompt contains any month name (June, July, Dec, December, etc.) AND CollectionHint is "dating", you MUST create a date range query on "Dating_Date" field.
- NEVER return empty object {} for dating collection when month is mentioned.
- Mandatory patterns for dating + month:
  * "datings in June" → { "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-06-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-06-30T23:59:59Z", "timezone": "UTC" } } } }
  * "datings in Dec 2025" → { "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-12-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-12-31T23:59:59Z", "timezone": "UTC" } } } }

Important: If the user's query is clearly about events or dating (not users), then for the users collection, return an empty query object {} (not a filter for all users). This ensures that the users search does not return all users when the query is not about users. For example, if the prompt is "Startup Pitch events in November", the users query should be {}.

CRITICAL DATING COLLECTION RULES:
- CollectionHint "dating" means you are generating a query for the "datings" collection.
- When CollectionHint is "dating" and the prompt contains month names (June, Dec, December, etc.), you MUST generate a date range query on the "Dating_Date" field.
- NEVER return {} for dating collection when a month is mentioned - always generate the proper date range.
- Example: If CollectionHint is "dating" and prompt is "Datings in Dec 2025", you MUST return: { "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-12-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-12-31T23:59:59Z", "timezone": "UTC" } } } }
- Treat "datings", "dating", "date", and "dates" as synonyms for the dating collection and its date field.
- ANY prompt mentioning "datings", "dating", "date", or "dates" followed by a month name should ALWAYS map to the "Dating_Date" field in the "datings" collection.
- MANDATORY: For queries like "datings in June", "dating in June", "dates in June", or "date in June", always generate a proper date range query on the "Dating_Date" field.
- Example: "datings in June" MUST produce: { "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-06-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-06-30T23:59:59Z", "timezone": "UTC" } } } }

Important global rules:
1. Always use the exact field names used by the collection:
   - Events collection: use "Event_date", "Event_location", "Event_type", etc.
   - Users collection: use "DOB" (date of birth) if the prompt references birthdate or age; other fields: "Name", "Gender", "Location", etc.
   - Datings collection: use "Dating_Date", "Dating_location" (if present), "Male_id", "Female_id", etc.

2. Text matching: ALWAYS use case-insensitive MongoDB operators for ANY text searches, including types, names, locations, etc. MANDATORY: Use "$regex" with "$options": "i" for ALL substring or partial matches in ANY text fields (e.g., Event_location, Event_type, Name, Location, etc.). NEVER use exact string matching for text fields—always partial with regex.

3. Gender matching (CRITICAL): For gender-related queries in users collection:
   - "male", "man", "men", "guy", "boy" → { "Gender": { "$regex": "^male$", "$options": "i" } }
   - "female", "woman", "women", "girl", "lady" → { "Gender": { "$regex": "^female$", "$options": "i" } }
   - Use EXACT match with anchors (^...$) for Gender field ONLY to avoid partial matches
   - Examples: "male in jaipur" → { "Gender": { "$regex": "^male$", "$options": "i" }, "Location": { "$regex": "jaipur", "$options": "i" } }
   Examples: 
   - { "Event_location": { "$regex": "mumbai", "$options": "i" } }
   - { "Event_type": { "$regex": "startup pitch", "$options": "i" } }
   - { "Name": { "$regex": "john", "$options": "i" } }

3. Dates and ranges (critical for all date fields like "Event_date", "Dating_Date", "DOB"):
   - NEVER use $expr, $month, $year, or any aggregation operators in the query. ALWAYS use simple $gte/$lte/$gt/$lt ranges with $dateFromString for ALL date queries. This is mandatory and more reliable.
   - Format for dates: { "$dateFromString": { "dateString": "YYYY-MM-DDTHH:mm:ssZ", "timezone": "UTC" } }
   - For exact day queries (e.g., "1st December 2025"): Use a range for that day: $gte start-of-day, $lte end-of-day.
     Example: { "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-12-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-12-01T23:59:59Z", "timezone": "UTC" } } } }
   - For month+year (e.g., "December 2025"): Use a range from first to last day of the month.
     Example: { "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-12-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-12-31T23:59:59Z", "timezone": "UTC" } } } }
   - For month-only (e.g., "events in November"): Infer year using YEAR-INFERENCE RULE and use month range. ALWAYS treat month names as dates, not locations.
     Example for November 2025: { "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-11-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-11-30T23:59:59Z", "timezone": "UTC" } } } }
   - For year-only (e.g., "events in 2025"): Use range from Jan 1 to Dec 31.
   - For relative expressions ("this month", "next month", "last month"): Calculate absolute dates using CurrentServerDate and use $gte/$lte ranges.
     Example: If CurrentServerDate is 2025-09-23 and prompt is "next month", use October 2025 range.
   - Apply to all date fields: "Event_date", "Dating_Date", "DOB" uniformly.
   - Handle typos in dates/months (e.g., "novemeber" -> "November", "decembre" -> "December").

4. YEAR-INFERENCE RULE (when user omits the year or parts):
   - Use "CurrentServerDate" from the user message to infer:
     * If requested month >= current month → use current year.
     * If requested month < current month → use next year.
     * For days: If day in future of current month/year, use current; if past, roll to next month/year as appropriate.
     * Example: CurrentServerDate 2025-09-23, "November" (11 > 9) → 2025-11.
     * Example: CurrentServerDate 2025-09-23, "August" (8 < 9) → 2026-08.
   - For ambiguous partial dates (e.g., "the 1st" without month), assume current month/year, or next plausible.
   - If CurrentServerDate not provided, assume current year from a default like 2025, but prefer {} if uncertain.

5. Age / DOB conversions:
   - For age (e.g., "aged 25 to 30"): Convert to DOB ranges using CurrentServerDate. Age 25 means DOB <= CurrentServerDate - 25 years, > CurrentServerDate - 26 years (for lower bound).
     Use $gte/$lt with $dateFromString for the calculated ISO dates. Calculate precisely: for min age A, DOB <= current - A years; for max age B, DOB > current - (B+1) years.
     Example (assume CurrentServerDate 2025-09-23): for ages 25-30, { "DOB": { "$lte": { "$dateFromString": { "dateString": "2000-09-23T00:00:00Z" } }, "$gt": { "$dateFromString": { "dateString": "1994-09-23T00:00:00Z" } } } } (adjust dates based on calc).
   - For "born in [YEAR]" (e.g., "born in 1995"): Create a date range for the entire year on the DOB field. For 1995, this would be from "1995-01-01T00:00:00Z" to "1995-12-31T23:59:59Z".
   - For "born in December" (no year): Use range for December of inferred year.
   - For "born in December 1990": Use month range with year.

6. Location vs. Time distinction (CRITICAL):
   - If the query contains "in [X]", determine if X is a time indicator (month, year, day, "next week", etc.) or a location (city, state, country).
   - Month names (January-December, Jan-Dec, typos like novemeber) or numbers (2025) after "in" MUST be treated as dates, using date ranges on the appropriate date field.
   - City/place names (e.g., Delhi, Mumbai, New York) after "in" MUST be treated as locations, using $regex on the location field.
   - Use context: "events in November" → date (month). "events in Delhi" → location. "events in Delhi in November" → both location and date.
   - NEVER misinterpret month names as locations—prioritize date if it matches a month.

7. Tolerant parsing and YEAR-INFERENCE:
  - Handle typos, abbreviations (e.g., "Dec", "1st", "first", "oct", "novemeber"→November), ordinals, formats (DD/MM/YYYY, MM/DD/YYYY—assume DD/MM if ambiguous, use context).
  - For ambiguity (e.g., "March" could be past/future), use YEAR-INFERENCE; if still ambiguous, return {"__ambiguous": true, "alternatives": [ {...}, {...} ] }.
  - For month-only queries (e.g., "events in August"), ALWAYS use the current year from CurrentServerDate unless the user specifies a year. Do NOT use a future year unless explicitly requested.

8. Output rules (CRITICAL):
   - Return ONLY a valid JSON object that can be parsed by JSON.parse()
   - Keys must be valid JSON strings quoted with double quotes
   - MongoDB operators like "$gte", "$lte", "$dateFromString" must be quoted with double quotes
   - Field names like "Event_date" must be quoted with double quotes
   - NEVER use $expr—always use $gte/$lte ranges for dates
   - Always use $dateFromString for date values, never plain strings or ISODate
   - For ALL text searches (locations, types, etc.), MANDATORY: use $regex with $options: "i"
   - If you cannot parse the prompt, return {}
   - EXAMPLE of correct output: { "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-11-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-11-30T23:59:59Z", "timezone": "UTC" } } } }

Additional Examples (follow these patterns strictly):
- Prompt: "events in november"
  Output: { "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-11-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-11-30T23:59:59Z", "timezone": "UTC" } } } }
- Prompt: "events in mumbai"
  Output: { "Event_location": { "$regex": "mumbai", "$options": "i" } }
- Prompt: "startup pitch events in delhi in novemeber"
  Output: { "Event_type": { "$regex": "startup pitch", "$options": "i" }, "Event_location": { "$regex": "delhi", "$options": "i" }, "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-11-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-11-30T23:59:59Z", "timezone": "UTC" } } } }
- Prompt: "male in jaipur"
  Output: { "Gender": { "$regex": "^male$", "$options": "i" }, "Location": { "$regex": "jaipur", "$options": "i" } }
- Prompt: "male born in 1995"
  Output: { "Gender": { "$regex": "^male$", "$options": "i" }, "DOB": { "$gte": { "$dateFromString": { "dateString": "1995-01-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "1995-12-31T23:59:59Z", "timezone": "UTC" } } } }
- Prompt: "female users aged 25 in jaipur"
  Output: { "Gender": { "$regex": "^female$", "$options": "i" }, "DOB": { "$lte": { "$dateFromString": { "dateString": "2000-09-23T00:00:00Z" } }, "$gt": { "$dateFromString": { "dateString": "1999-09-23T00:00:00Z" } } }, "Location": { "$regex": "jaipur", "$options": "i" } }
- Prompt: "datings in june"
  Output: { "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-06-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-06-30T23:59:59Z", "timezone": "UTC" } } } }
- Prompt: "dating in december"
  Output: { "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-12-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-12-31T23:59:59Z", "timezone": "UTC" } } } }
- Prompt: "datings in bengaluru next month"
  Output: { "Dating_location": { "$regex": "bengaluru", "$options": "i" }, "Dating_Date": { "$gte": { "$dateFromString": { "dateString": "2025-10-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-10-31T23:59:59Z", "timezone": "UTC" } } } }

`;

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
// Recursively convert $dateFromString in query objects to JS Date objects
function convertDateFromString(obj) {
  if (Array.isArray(obj)) {
    return obj.map(convertDateFromString);
  } else if (obj && typeof obj === "object") {
    if ("$dateFromString" in obj && obj["$dateFromString"].dateString) {
      return new Date(obj["$dateFromString"].dateString);
    }
    const out = {};
    for (const key in obj) {
      out[key] = convertDateFromString(obj[key]);
    }
    return out;
  }
  return obj;
}
function hideSensitive(doc) {
  if (!ALLOW_PII) {
    const { Salary, email, ...rest } = doc;
    return rest;
  }
  return doc;
}

// ----- Dynamic prompt-to-query parser -----
async function parsePromptToMongoQuery(prompt, type) {
  if (!openai || !prompt) return {};
  const currentServerDate = new Date().toISOString();
  const systemMsg = SYSTEM_PROMPT;
  const userMsg = `CurrentServerDate: ${currentServerDate}
CollectionHint: ${type}
Prompt: ${prompt}`;
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemMsg },
        { role: "user", content: userMsg },
      ],
      temperature: 0.0,
      max_tokens: 1500,
    });
    const text = completion.choices[0].message.content.trim();
    console.log(
      `[DEBUG] OpenAI response for ${type} collection with prompt "${prompt}":`,
      text
    );

    try {
      const parsed = JSON.parse(text);
      console.log(
        `[DEBUG] Parsed query for ${type}:`,
        JSON.stringify(parsed, null, 2)
      );
      return parsed;
    } catch {
      console.log(
        `[DEBUG] Failed to parse query for ${type}, returning empty object`
      );
      return {};
    }
  } catch (err) {
    console.error("NLP parse error:", err);
    return {};
  }
}

// ----- MongoDB connection & server start -----
let db;
let client;

async function startServer() {
  client = new MongoClient(MONGO_URI);
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
      let filter = await parsePromptToMongoQuery(q, "users");
      filter = convertDateFromString(filter);
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
      const populate = req.query.populate === 'true';
      const l = Math.min(Number(limit) || 10, 100);
      const baseFilter = await parsePromptToMongoQuery(q, "events");
      const filter = convertDateFromString(baseFilter);
      const cursor = db.collection("events").find(filter).limit(l);
      let docs = await cursor.toArray();

      if (populate && docs.length) {
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
      const populate = req.query.populate === 'true';
      const l = Math.min(Number(limit) || 10, 100);
      const baseFilter = await parsePromptToMongoQuery(q, "dating");
      const filter = convertDateFromString(baseFilter);

      let docs = await db
        .collection("datings")
        .find(filter)
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

  // SEARCH ALL: Search across all three collections
  app.post("/search/all", async (req, res) => {
    try {
      const { q = "", limit = 10 } = req.body || {};
      const populate = req.query.populate === 'true';
      const l = Math.min(Number(limit) || 10, 30); // Reduced per-collection limit for combined search

      // Search all three collections in parallel
      const [usersFilter, eventsFilter, datingFilter] = await Promise.all([
        parsePromptToMongoQuery(q, "users"),
        parsePromptToMongoQuery(q, "events"),
        parsePromptToMongoQuery(q, "dating"),
      ]);

      // Only run queries if the filter is not empty
      let usersResults = [];
      if (usersFilter && Object.keys(usersFilter).length > 0) {
        usersResults = await db
          .collection("users")
          .find(convertDateFromString(usersFilter), {
            projection: ALLOW_PII ? {} : { Salary: 0, email: 0 },
          })
          .limit(l)
          .toArray();
        usersResults = usersResults.map(hideSensitive);
      }

      let eventsResults = [];
      if (eventsFilter && Object.keys(eventsFilter).length > 0) {
        eventsResults = await db
          .collection("events")
          .find(convertDateFromString(eventsFilter))
          .limit(l)
          .toArray();
      }

      let datingResults = [];
      if (datingFilter && Object.keys(datingFilter).length > 0) {
        datingResults = await db
          .collection("datings")
          .find(convertDateFromString(datingFilter))
          .limit(l)
          .toArray();
      }

      // Populate if requested
      if (populate) {
        // Populate events participants
        if (eventsResults.length) {
          const eventUserIds = Array.from(
            new Set(
              eventsResults.flatMap((d) =>
                (d.participant_ids || []).map((id) => id.toString())
              )
            )
          ).map((s) => new ObjectId(s));

          if (eventUserIds.length) {
            const eventUsers = await db
              .collection("users")
              .find(
                { _id: { $in: eventUserIds } },
                { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
              )
              .toArray();
            const eventUsersById = new Map(
              eventUsers.map((u) => [u._id.toString(), hideSensitive(u)])
            );

            eventsResults.forEach((ev) => {
              const participants = (ev.participant_ids || []).map(
                (id) => eventUsersById.get(id.toString()) || { _id: id }
              );
              ev.participants = participants;
            });
          }
        }

        // Populate dating Male/Female
        if (datingResults.length) {
          const datingUserIds = Array.from(
            new Set(
              datingResults.flatMap((d) =>
                [d.Male_id, d.Female_id]
                  .filter(Boolean)
                  .map((id) => id.toString())
              )
            )
          ).map((s) => new ObjectId(s));

          if (datingUserIds.length) {
            const datingUsers = await db
              .collection("users")
              .find(
                { _id: { $in: datingUserIds } },
                { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
              )
              .toArray();
            const datingUsersById = new Map(
              datingUsers.map((u) => [u._id.toString(), hideSensitive(u)])
            );

            datingResults.forEach((dt) => {
              dt.Male = dt.Male_id
                ? datingUsersById.get(dt.Male_id.toString())
                : null;
              dt.Female = dt.Female_id
                ? datingUsersById.get(dt.Female_id.toString())
                : null;
            });
          }
        }
      }

      const totalCount =
        usersResults.length + eventsResults.length + datingResults.length;

      return res.json({
        count: totalCount,
        results: {
          users: { count: usersResults.length, data: usersResults },
          events: { count: eventsResults.length, data: eventsResults },
          dating: { count: datingResults.length, data: datingResults },
        },
        queries: {
          users: usersFilter,
          events: eventsFilter,
          dating: datingFilter,
        },
      });
    } catch (err) {
      console.error("search/all error", err);
      return res.status(500).json({ error: "Server error" });
    }
  });

  // Start Express
  app.listen(PORT, () => console.log(`API listening http://localhost:${PORT}`));
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
        {
          name: "search_all",
          description:
            "Search across all collections (users, events, dating) with plain English queries",
          inputSchema: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "Plain English search query for all collections",
              },
              limit: {
                type: "number",
                description:
                  "Maximum number of results per collection (default: 10, max: 30)",
                default: 10,
              },
              populate: {
                type: "boolean",
                description: "Include full user details for related records",
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
          let filter = await parsePromptToMongoQuery(query, "users");
          filter = convertDateFromString(filter);
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
          let baseFilter = await parsePromptToMongoQuery(query, "events");
          baseFilter = convertDateFromString(baseFilter);
          let results = await db
            .collection("events")
            .find(baseFilter)
            .limit(l)
            .toArray();

          if (populate) {
            for (let event of results) {
              if (
                event.participant_ids &&
                Array.isArray(event.participant_ids)
              ) {
                const userIds = event.participant_ids.map((p) =>
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
                    query: baseFilter,
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
          let baseFilter = await parsePromptToMongoQuery(query, "dating");
          baseFilter = convertDateFromString(baseFilter);
          let results = await db
            .collection("datings")
            .find(baseFilter)
            .limit(l)
            .toArray();

          if (populate) {
            for (let profile of results) {
              if (profile.Male_id) {
                const maleId =
                  typeof profile.Male_id === "string"
                    ? new ObjectId(profile.Male_id)
                    : profile.Male_id;
                const male = await db
                  .collection("users")
                  .findOne(
                    { _id: maleId },
                    { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
                  );
                profile.maleDetails = male;
              }
              if (profile.Female_id) {
                const femaleId =
                  typeof profile.Female_id === "string"
                    ? new ObjectId(profile.Female_id)
                    : profile.Female_id;
                const female = await db
                  .collection("users")
                  .findOne(
                    { _id: femaleId },
                    { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
                  );
                profile.femaleDetails = female;
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
                    query: baseFilter,
                    populated: populate,
                  },
                  null,
                  2
                ),
              },
            ],
          };
        }

        case "search_all": {
          const { query = "", limit = 10, populate = false } = args;
          const l = Math.min(Number(limit) || 10, 30);

          // Search all three collections in parallel
          const [usersFilter, eventsFilter, datingFilter] = await Promise.all([
            parsePromptToMongoQuery(query, "users"),
            parsePromptToMongoQuery(query, "events"),
            parsePromptToMongoQuery(query, "dating"),
          ]);

          const [usersResults, eventsResults, datingResults] =
            await Promise.all([
              db
                .collection("users")
                .find(convertDateFromString(usersFilter), {
                  projection: ALLOW_PII ? {} : { Salary: 0, email: 0 },
                })
                .limit(l)
                .toArray(),
              db
                .collection("events")
                .find(convertDateFromString(eventsFilter))
                .limit(l)
                .toArray(),
              db
                .collection("datings")
                .find(convertDateFromString(datingFilter))
                .limit(l)
                .toArray(),
            ]);

          if (populate) {
            // Populate events
            for (let event of eventsResults) {
              if (
                event.participant_ids &&
                Array.isArray(event.participant_ids)
              ) {
                const userIds = event.participant_ids.map((p) =>
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

            // Populate dating
            for (let profile of datingResults) {
              if (profile.Male_id) {
                const male = await db.collection("users").findOne(
                  {
                    _id:
                      typeof profile.Male_id === "string"
                        ? new ObjectId(profile.Male_id)
                        : profile.Male_id,
                  },
                  { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
                );
                profile.maleDetails = male;
              }
              if (profile.Female_id) {
                const female = await db.collection("users").findOne(
                  {
                    _id:
                      typeof profile.Female_id === "string"
                        ? new ObjectId(profile.Female_id)
                        : profile.Female_id,
                  },
                  { projection: ALLOW_PII ? {} : { Salary: 0, email: 0 } }
                );
                profile.femaleDetails = female;
              }
            }
          }

          const totalCount =
            usersResults.length + eventsResults.length + datingResults.length;

          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(
                  {
                    totalCount,
                    results: {
                      users: { count: usersResults.length, data: usersResults },
                      events: {
                        count: eventsResults.length,
                        data: eventsResults,
                      },
                      dating: {
                        count: datingResults.length,
                        data: datingResults,
                      },
                    },
                    queries: {
                      users: usersFilter,
                      events: eventsFilter,
                      dating: datingFilter,
                    },
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