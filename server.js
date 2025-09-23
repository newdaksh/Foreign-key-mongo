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
const PORT = 8001;

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
   - Datings collection: use "Dating_date", "Dating_location" (if present), "Male_id", "Female_id", etc.

2. Text matching: ALWAYS use case-insensitive MongoDB operators for ANY text searches, including types, names, locations, etc. MANDATORY: Use "$regex" with "$options": "i" for ALL substring or partial matches in ANY text fields (e.g., Event_location, Event_type, Name, Location, etc.). NEVER use exact string matching for text fields—always partial with regex.
   Examples: 
   - { "Event_location": { "$regex": "mumbai", "$options": "i" } }
   - { "Event_type": { "$regex": "startup pitch", "$options": "i" } }
   - { "Name": { "$regex": "john", "$options": "i" } }

3. Dates and ranges (critical for all date fields like "Event_date", "Dating_date", "DOB"):
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
   - Apply to all date fields: "Event_date", "Dating_date", "DOB" uniformly.
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
- Prompt: "users aged 25 in jaipur"
  Output: { "DOB": { "$lte": { "$dateFromString": { "dateString": "2000-09-23T00:00:00Z" } }, "$gt": { "$dateFromString": { "dateString": "1999-09-23T00:00:00Z" } } }, "Location": { "$regex": "jaipur", "$options": "i" } }
- Prompt: "datings in bengaluru next month"
  Output: { "Dating_location": { "$regex": "bengaluru", "$options": "i" }, "Dating_date": { "$gte": { "$dateFromString": { "dateString": "2025-10-01T00:00:00Z", "timezone": "UTC" } }, "$lte": { "$dateFromString": { "dateString": "2025-10-31T23:59:59Z", "timezone": "UTC" } } } }

End of system rules. REMEMBER: output must be strictly a valid JSON object for MongoDB query parsing.
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

    // Sanitize OpenAI output: only replace field names like $Event_date, not MongoDB operators
    let sanitized = raw
      // Replace field names starting with $ (but not MongoDB operators)
      .replace(
        /\$(?!gte|lte|eq|gt|lt|ne|in|nin|and|or|not|nor|exists|type|regex|options|size|all|elemMatch|where|expr|jsonSchema|mod|text|search|language|caseSensitive|diacriticSensitive|near|nearSphere|geoWithin|geoIntersects|geometry|maxDistance|minDistance|center|centerSphere|box|polygon|uniqueDocs|inc|mul|rename|setOnInsert|set|unset|min|max|currentDate|addToSet|pop|pullAll|pull|pushAll|push|each|slice|sort|position|bit|isolated|dateFromString|month|year|dayOfMonth|date|timezone)([A-Za-z_][A-Za-z0-9_]*)/g,
        '"$1"'
      )
      // Quote unquoted field names (best effort)
      .replace(/([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)/g, '$1"$2"$3');

    try {
      const parsed = JSON.parse(sanitized);
      return parsed;
    } catch (e) {
      // If still can't parse, try the original raw string
      try {
        const parsed = JSON.parse(raw.trim());
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
 * Body: { query: "Events in October", limit: 10 } // searchType is optional, auto-detected
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

    // Improved auto-detect searchType if not provided
    if (!searchType) {
      const lowerPrompt = prompt.toLowerCase();
      if (
        lowerPrompt.match(
          /\b(event|meetup|conference|workshop|gathering|pitch|startup|party|seminar)\b/
        )
      ) {
        searchType = "events";
      } else if (
        lowerPrompt.match(
          /\b(datings|match|relationship|couple|date|romantic|pairing)\b/
        )
      ) {
        searchType = "datings";
      } else if (
        lowerPrompt.match(
          /\b(male|female|user|person|people|member|profile|age|gender|location|name|boy|girl|man|woman|jaipur|delhi|mumbai|bengaluru|chennai)\b/
        )
      ) {
        searchType = "users";
      } else {
        searchType = "users"; // default fallback is users
      }
    }

    // Map searchType -> collection name
    const collectionMap = {
      events: "events",
      users: "users",
      datings: "datings",
    };
    const collectionName = collectionMap[searchType] || "users";

    // Ask the model to produce a valid MongoJSON query
    let mongoQuery = await parsePromptToMongoQuery(prompt, collectionName);
    mongoQuery = convertDateFromString(mongoQuery);

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
    if (collectionName === "datings" && results && results.length > 0) {
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
      'Use POST /search/debug with body { "query": "your search" }' // searchType optional
    );
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
