const mongoose = require("mongoose");

const connectDB = async () => {
  try {
    // Check if MONGO_URI is properly configured
    if (!process.env.MONGO_URI || process.env.MONGO_URI.includes('your-actual-mongodb')) {
      console.log("⚠️  MongoDB URI not configured. Using local file storage instead.");
      console.log("📝 To use MongoDB, update MONGO_URI in .env file with your actual connection string.");
      return;
    }

    await mongoose.connect(process.env.MONGO_URI);
    console.log("✅ MongoDB Connected successfully!");
  } catch (error) {
    console.error("❌ MongoDB connection failed:", error.message);
    console.log("📝 Continuing without database. Update MONGO_URI in .env to connect to MongoDB.");
    // Don't exit, continue without database
  }
};

module.exports = connectDB;
