const express = require("express");
const multer = require("multer");
const loadModel = require("./loadModel");
const crypto = require("crypto");
const predictClassification = require("./inferenceService");

const app = express();
const port = 3000;

const upload = multer({
  limits: { fileSize: 1000000 },
});

app.post("/predict", upload.single("image"), async (req, res) => {
  const imageFile = req.file;
  const id = crypto.randomUUID();
  const model = await loadModel();
  try {
    const { confidenceScore } = await predictClassification(model, imageFile);
    const predictionResult = {
      id: id,
      result: confidenceScore > 0.5 ? "Cancer" : "Non-Cancer",
      suggestion:
        confidenceScore > 0.5
          ? "Segera periksa ke dokter!"
          : "Tidak kena kanker",
      createdAt: new Date().toISOString(),
    };
    res.status(200).json({
      status: "success",
      message: "Model is predicted successfully",
      data: predictionResult,
    });
  } catch (error) {
    res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
      data: predictionResult,
    });
  }
});

app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
    // Jika terjadi kesalahan batas ukuran file
    return res.status(413).json({
      status: "fail",
      message: "Payload content length greater than maximum allowed: 1000000",
    });
  }
  next();
});

// Jalankan server pada port yang ditentukan
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
