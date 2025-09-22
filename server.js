// server.js
import express from "express";
import bodyParser from "body-parser";
import dotenv from "dotenv";
import { MongoClient } from "mongodb";
import OpenAI from "openai";

dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.warn("WARNING: OPENAI_API_KEY not set. OpenAI calls will fail.");
}
const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017";
const DB_NAME = process.env.DB_NAME || "mydb";
const PORT = process.env.PORT ? Number(process.env.PORT) : 8000;

const app = express();
app.use(bodyParser.json());

// Mongo client & DB placeholder
let mongoClient;
let db;

async function initDB() {
  mongoClient = new MongoClient(MONGO_URI);
  await mongoClient.connect();
  db = mongoClient.db(DB_NAME);
  console.log("Connected to MongoDB:", MONGO_URI, "DB:", DB_NAME);
}

/**
 * Strict system prompt for model to convert user's natural-language to a valid JSON Mongo query.
 * IMPORTANT: The model is instructed to output ONLY a JSON object and nothing else.
 */
const SYSTEM_PROMPT = `
SYSTEM ROLE: You are an assistant whose only job is to convert a user's natural-language search prompt into a single valid JSON object that is a MongoDB query for a specific collection. RETURN ONLY A JSON OBJECT (no explanation, no extra text, no code fences). The output MUST be a syntactically valid JSON object that MongoDB accepts when parsed.

Important global rules:
1. Always use the exact field names used by the collection:
   - Events collection: use "Event_date", "Event_location", "Event_type", etc.
   - Users collection: use "DOB" (date of birth) if the prompt references birthdate or age; other fields: "Name", "Gender", "Location", etc.
   - Dating collection: use "Dating_date", "Dating_location" (if present), "Male_id", "Female_id", etc.

2. Text matching: ALWAYS use case-insensitive MongoDB operators for ANY text/location searches. MANDATORY: Use "$regex" with "$options": "i" for ALL substring matches in location fields. 
   Examples: 
   - { "Event_location": { "$regex": "mumbai", "$options": "i" } }
   - { "Dating_location": { "$regex": "bengaluru", "$options": "i" } }
   - { "Location": { "$regex": "chennai", "$options": "i" } }

3. Dates and ranges (critical for all date fields like "Event_date", "Dating_date", "DOB"):
   - Always use MongoDB's $dateFromString operator to specify any date literals in comparisons to ensure proper Date type handling. Do not use plain strings for date values.
   - Format: { "$dateFromString": { "dateString": "YYYY-MM-DDTHH:mm:ssZ", "timezone": "UTC" } }. Omit seconds/millis if not needed, but prefer full ISO for precision.
   - For exact day queries (e.g., "1st December 2025"): Use a range for that day: $gte start-of-day (T00:00:00Z), $lte end-of-day (T23:59:59Z), using $dateFromString for both.
     Example: { "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-12-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-12-01T23:59:59Z", "timezone": "UTC" } } } }
   - For month+year (e.g., "December 2025"): Use a range from first to last day of the month, with $dateFromString.
     Example: { "Event_date": { "$gte": { "$dateFromString": { "dateString": "2025-12-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-12-31T23:59:59Z", "timezone": "UTC" } } } }
   - For month-only (e.g., "events in October" or "born in December"): Infer year using YEAR-INFERENCE RULE (below), then use month range as above. If no specific year and searching across years, use $expr with $month (and optionally $year if bounded).
     Example for any December: { "$expr": { "$eq": [ { "$month": { "date": "$DOB", "timezone": "UTC" } }, 12 ] } }
   - For date-only or partial dates (e.g., "events on the 1st", "birthdays on 15th"): Infer month/year using YEAR-INFERENCE RULE and CurrentServerDate, then use $expr for day/month/year matching.
     Example: { "$expr": { "$and": [ { "$eq": [ { "$dayOfMonth": { "date": "$Event_date", "timezone": "UTC" } }, 1 ] }, { "$eq": [ { "$month": { "date": "$Event_date", "timezone": "UTC" } }, 12 ] }, { "$eq": [ { "$year": { "date": "$Event_date", "timezone": "UTC" } }, 2025 ] } ] } }
   - For relative expressions ("this month", "next month", "today", "tomorrow", "yesterday", "next year"): Calculate absolute dates using the provided CurrentServerDate. Use date math logically (e.g., "next month" = current month +1, handle year rollover).
   - Apply to all date fields: "Event_date", "Dating_date", "DOB" uniformly.

4. YEAR-INFERENCE RULE (when user omits the year or parts):
   - Use "CurrentServerDate" from the user message to infer:
     * If requested month >= current month → use current year.
     * If requested month < current month → use next year.
     * For days: If day in future of current month/year, use current; if past, roll to next month/year as appropriate.
   - For ambiguous partial dates (e.g., "the 1st" without month), assume current month/year, or next plausible.
   - If CurrentServerDate not provided, assume current year from a default like 2025, but prefer {} if uncertain.

5. Age / DOB conversions:
   - For age (e.g., "aged 25 to 30"): Convert to DOB ranges using CurrentServerDate. Age 25 means DOB <= CurrentServerDate - 25 years, > CurrentServerDate - 26 years (for lower bound).
     Use $gte/$lt with $dateFromString for the calculated ISO dates.
     Example: { "DOB": { "$lte": { "$dateFromString": { "dateString": "2000-09-22T00:00:00Z" } }, "$gt": { "$dateFromString": { "dateString": "1995-09-22T00:00:00Z" } } } } (adjust dates based on calc).
   - For "born in December" (no year): Use $expr with $month as above.
   - For "born in December 1990": Use month range with year as above.

6. Location matching rules (CRITICAL):
   - For location searches, ALWAYS use $regex with $options: "i" for case-insensitive partial matching
   - Dating collection: When user searches "datings in [city]", use: { "Dating_location": { "$regex": "[city]", "$options": "i" } }
   - Events collection: When user searches "events in [city]", use: { "Event_location": { "$regex": "[city]", "$options": "i" } }
   - Users collection: When user searches "users in [city]", use: { "Location": { "$regex": "[city]", "$options": "i" } }
   - NEVER use exact string matching for locations - always use regex for partial matches

7. Tolerant parsing:
   - Handle typos, abbreviations (e.g., "Dec", "1st", "first", "oct"), ordinals, formats (DD/MM/YYYY, MM/DD/YYYY—assume DD/MM if ambiguous, use context).
   - For ambiguity (e.g., "March" could be past/future), use YEAR-INFERENCE; if still ambiguous, return {"__ambiguous": true, "alternatives": [ {...}, {...} ] }.

8. Output rules:
   - Return only one JSON object (no explanation).
   - Keys must be valid JSON strings and Mongo operators quoted (e.g., "$gte", "$expr").
   - Use $and / $or / $expr only when needed for complex logic.
   - Always incorporate timezone: "UTC" in $dateFromString or date functions to avoid issues.
   - For location searches, MANDATORY: use $regex with $options: "i" - never use exact string matching.
   - If you cannot parse the prompt, return {}.

End of system rules. REMEMBER: output must be strictly a valid JSON object for a MongoDB query and nothing else.
`;

/**
 * Calls the OpenAI Chat completion with the strict system prompt and a user message
 * that includes CurrentServerDate for year inference. The model is expected to return
 * only a JSON object (as text). We parse it and return the object. If parse fails,
 * return {}.
 */
async function parsePromptToMongoQuery(prompt, collectionHint = "events") {
  if (!openai) {
    console.error("OpenAI not configured; cannot parse prompt to query.");
    return {};
  }

  const currentServerDate = new Date().toISOString();
  // User message includes server date (for year inference) + optional collection hint + original prompt
  const userMessage = `CurrentServerDate: ${currentServerDate}
CollectionHint: ${collectionHint}
Prompt: ${prompt}`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini", // if your account uses a different model, change here
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userMessage },
      ],
      temperature: 0.0,
      max_tokens: 1500,
    });

    // Access text. SDKs differ; handle common shape:
    const raw =
      response?.choices?.[0]?.message?.content ?? response?.choices?.[0]?.text;
    if (!raw) {
      console.error("OpenAI returned no text content:", response);
      return {};
    }

    // Try to parse raw as JSON directly (model is instructed to return only JSON)
    try {
      const parsed = JSON.parse(raw);
      return parsed;
    } catch (e) {
      // If direct parsing fails, do NOT do aggressive regex fixes (per user's preference).
      // Minimal graceful attempt: trim surrounding whitespace and try again; otherwise return {}
      const trimmed = raw.trim();
      try {
        const parsed = JSON.parse(trimmed);
        return parsed;
      } catch (e2) {
        console.error(
          "Unable to parse assistant output as JSON. Raw output logged below:"
        );
        console.error(raw);
        return {};
      }
    }
  } catch (err) {
    console.error("OpenAI API error:", err);
    return {};
  }
}

/**
 * Helper: run a Mongo query safely. If query is empty object, return empty results.
 * This function will not try to mutate or interpret the query — it uses it as-is.
 */
async function runMongoQuery(collectionName, mongoQuery, limit = 50) {
  if (!db) throw new Error("DB not initialized");
  const coll = db.collection(collectionName);
  // If the query is an empty object {}, we return nothing
  if (
    !mongoQuery ||
    (Object.keys(mongoQuery).length === 0 && mongoQuery.constructor === Object)
  ) {
    return { count: 0, results: [] };
  }

  // Defensive: ensure limit is reasonable
  const cappedLimit = Math.min(Math.max(limit, 1), 1000);

  // Execute query
  const cursor = coll.find(mongoQuery).limit(cappedLimit);
  const results = await cursor.toArray();
  return { count: results.length, results };
}

/**
 * Main search endpoint used in screenshots: POST /search/debug
 * Body: { query: "Events in October", searchType: "events|users|dating", limit: 10 }
 */
app.post("/search/debug", async (req, res) => {
  try {
    const body = req.body || {};
    const prompt = (body.query || "").toString();
    let searchType = body.searchType
      ? body.searchType.toString().toLowerCase()
      : "";
    const limit = Number.isInteger(body.limit) ? body.limit : 10;

    if (!prompt) {
      return res
        .status(400)
        .json({ error: "Missing 'query' in request body." });
    }

    // Auto-detect searchType if not provided
    if (!searchType) {
      const lowerPrompt = prompt.toLowerCase();
      if (lowerPrompt.match(/event|meetup|conference|workshop|gathering/)) {
        searchType = "events";
      } else if (
        lowerPrompt.match(
          /user|person|people|member|profile|age|gender|location|name/
        )
      ) {
        searchType = "users";
      } else if (lowerPrompt.match(/dating|match|relationship|couple|date/)) {
        searchType = "dating";
      } else {
        searchType = "events"; // default fallback
      }
    }

    // Map searchType -> collection name
    const collectionMap = {
      events: "events",
      users: "users",
      dating: "dating",
    };
    const collectionName = collectionMap[searchType] || "events";

    // Ask the model to produce a valid MongoJSON query
    let mongoQuery = await parsePromptToMongoQuery(prompt, collectionName);

    // Force $options: 'i' for all $regex location fields
    function enforceRegexOptions(obj) {
      if (obj && typeof obj === "object") {
        for (const key of Object.keys(obj)) {
          if (typeof obj[key] === "object" && obj[key] !== null) {
            // Check for $regex in location fields
            if (
              (key.toLowerCase().includes("location") ||
                key.toLowerCase().includes("place")) &&
              obj[key].$regex
            ) {
              obj[key].$options = "i";
            }
            enforceRegexOptions(obj[key]);
          }
        }
      }
    }
    enforceRegexOptions(mongoQuery);

    // If model returned an __ambiguous object, include it in the response and do not run DB query
    if (mongoQuery && mongoQuery.__ambiguous) {
      return res.json({
        searchType: collectionName,
        query: mongoQuery,
        count: 0,
        results: [],
        note: "Ambiguous query returned by model. Please pick one alternative.",
      });
    }

    // Run the query against the chosen collection
    const { count, results } = await runMongoQuery(
      collectionName,
      mongoQuery,
      limit
    );

    // Enrich event results with full participant user documents
    if (collectionName === "events" && results && results.length > 0) {
      for (const event of results) {
        if (
          Array.isArray(event.participant_ids) &&
          event.participant_ids.length > 0
        ) {
          try {
            // Convert string IDs to ObjectId if needed
            const ids = event.participant_ids.map((id) => {
              if (typeof id === "string" && id.length === 24) {
                try {
                  return new mongoClient.constructor.ObjectId(id);
                } catch {
                  return id;
                }
              }
              return id;
            });
            const users = await db
              .collection("users")
              .find({ _id: { $in: ids } })
              .toArray();
            event.participants = users;
          } catch (e) {
            event.participants = [];
            console.error("Participant enrichment error:", e);
          }
        } else {
          event.participants = [];
        }
      }
    }
    // Enrich dating results with full user documents for Male_id and Female_id
    if (collectionName === "dating" && results && results.length > 0) {
      for (const doc of results) {
        try {
          if (doc.Male_id) {
            let maleId = doc.Male_id;
            if (typeof maleId === "string" && maleId.length === 24) {
              try {
                maleId = new mongoClient.constructor.ObjectId(maleId);
              } catch {}
            }
            const male = await db.collection("users").findOne({ _id: maleId });
            doc.male = male;
          }
          if (doc.Female_id) {
            let femaleId = doc.Female_id;
            if (typeof femaleId === "string" && femaleId.length === 24) {
              try {
                femaleId = new mongoClient.constructor.ObjectId(femaleId);
              } catch {}
            }
            const female = await db
              .collection("users")
              .findOne({ _id: femaleId });
            doc.female = female;
          }
        } catch (e) {
          // ignore enrichment errors
          console.error("Enrichment error:", e);
        }
      }
    }

    return res.json({
      searchType: collectionName,
      query: mongoQuery,
      count,
      results,
    });
  } catch (err) {
    console.error("/search/debug error:", err);
    return res
      .status(500)
      .json({ error: "Internal server error", details: err.message });
  }
});

/**
 * Optional helper endpoint to test system prompt parsing only (doesn't hit DB)
 * POST /parse/only { query: "Events in October", searchType: "events" }
 */
app.post("/parse/only", async (req, res) => {
  try {
    const prompt = (req.body?.query || "").toString();
    const searchType = (req.body?.searchType || "events").toString();
    if (!prompt) return res.status(400).json({ error: "Missing query" });

    const mongoQuery = await parsePromptToMongoQuery(prompt, searchType);
    return res.json({ query: mongoQuery });
  } catch (e) {
    console.error("/parse/only error:", e);
    return res.status(500).json({ error: e.message });
  }
});

// graceful shutdown
async function shutdown() {
  console.log("Shutting down...");
  try {
    if (mongoClient) await mongoClient.close();
  } catch (e) {
    console.error("Error closing Mongo client:", e);
  }
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

// Start server
async function startServer() {
  await initDB();
  app.listen(PORT, () => {
    console.log(`Smart MCP test server running at http://localhost:${PORT}`);
    console.log(
      'Use POST /search/debug with body { "query": "your search", "searchType": "events|users|dating" }'
    );
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
