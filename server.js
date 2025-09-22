import express from "express";
import bodyParser from "body-parser";
import { MongoClient, ObjectId } from "mongodb";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

const app = express();
app.use(bodyParser.json());

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017";
const DB_NAME = process.env.DB_NAME || "mydb";
const ALLOW_PII = process.env.ALLOW_PII === "true";

let db;
let client;

// Initialize MongoDB connection
async function initDB() {
  client = new MongoClient(MONGO_URI);
  await client.connect();
  db = client.db(DB_NAME);
  console.log("Connected to MongoDB for MCP testing");
}

// Smart query detection and execution
app.post("/search/debug", async (req, res) => {
  try {
    const query = req.body.query || "";
    // Detect what type of search based on keywords
    let searchType = "users"; // default
    const lowerQuery = query.toLowerCase();
    if (
      lowerQuery.includes("event") ||
      lowerQuery.includes("meetup") ||
      lowerQuery.includes("conference") ||
      lowerQuery.includes("gathering")
    ) {
      searchType = "events";
    } else if (
      lowerQuery.includes("dating") ||
      lowerQuery.includes("match") ||
      lowerQuery.includes("relationship") ||
      lowerQuery.includes("couple")
    ) {
      searchType = "datings";
    }

    let results = [];
    let mongoQuery = {};
    if (searchType === "users") {
      mongoQuery = await buildFilterFromQueryForUsers(query);
      results = await searchUsersWithQuery(mongoQuery);
    } else if (searchType === "events") {
      mongoQuery = await buildFilterFromQueryForEvents(query);
      results = await searchEventsWithQuery(mongoQuery);
    } else if (searchType === "datings") {
      mongoQuery = await buildFilterFromQueryForDating(query);
      results = await searchDatingWithQuery(mongoQuery);
    }
    res.json({
      searchType: searchType,
      query: query,
      mongoQuery,
      count: results.length,
      results: results,
    });
  } catch (error) {
    console.error("Search error:", error);
    res.status(500).json({ error: "Search failed", message: error.message });
  }
});

// Search functions
// New search functions using direct MongoDB query
async function searchUsersWithQuery(query) {
  const projection = ALLOW_PII ? {} : { Salary: 0, email: 0 };
  const results = await db
    .collection("users")
    .find(query, { projection })
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

async function searchEventsWithQuery(query) {
  const results = await db.collection("events").find(query).limit(10).toArray();
  for (let event of results) {
    if (event.participant_ids && Array.isArray(event.participant_ids)) {
      const participants = await db
        .collection("users")
        .find(
          { _id: { $in: event.participant_ids } },
          { projection: { Name: 1, Location: 1, Occupation: 1 } }
        )
        .toArray();
      event.participants = participants;
    }
  }
  return results.map((event) => ({
    eventType: event.Event_type,
    location: event.Event_location,
    participants: event.participants || [],
  }));
}

async function searchDatingWithQuery(query) {
  const results = await db
    .collection("datings")
    .find(query)
    .limit(10)
    .toArray();
  for (let profile of results) {
    if (profile.Male_id) {
      const male = await db
        .collection("users")
        .findOne(
          { _id: profile.Male_id },
          { projection: { Name: 1, age: 1, Location: 1, Occupation: 1 } }
        );
      profile.male = male;
    }
    if (profile.Female_id) {
      const female = await db
        .collection("users")
        .findOne(
          { _id: profile.Female_id },
          { projection: { Name: 1, age: 1, Location: 1, Occupation: 1 } }
        );
      profile.female = female;
    }
  }
  return results.map((datings) => ({
    location: datings.Dating_location,
    male: datings.male,
    female: datings.female,
  }));
}

// Filter building functions (copied from index.js)
async function buildFilterFromQueryForUsers(q) {
  return await parsePromptToMongoQuery(q, "users");
}

async function buildFilterFromQueryForEvents(q) {
  return await parsePromptToMongoQuery(q, "events");
}

async function buildFilterFromQueryForDating(q) {
  return await parsePromptToMongoQuery(q, "dating");
}
// ----- Dynamic prompt-to-query parser -----
async function parsePromptToMongoQuery(prompt, type) {
  if (!openai || !prompt) return {};
  const systemMsg = `You are an assistant that converts natural language search prompts into MongoDB query objects for the ${type} collection. Only return a valid MongoDB query object, no explanation.`;
  const userMsg = `Prompt: ${prompt}`;
  try {
    const completion = await openai.createChatCompletion({
      model: "gpt-3.5-turbo",
      messages: [
        { role: "system", content: systemMsg },
        { role: "user", content: userMsg },
      ],
      temperature: 0.2,
      max_tokens: 300,
    });
    // Extract JSON from response
    const text = completion.data.choices[0].message.content.trim();
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
}

// Start the server
async function startServer() {
  await initDB();
  app.listen(8000, () => {
    console.log("Smart MCP test server running at http://localhost:8000");
    console.log('Use POST /search/debug with { "query": "your search" }');
  });
}

startServer().catch(console.error);
