const tf = require("@tensorflow/tfjs-node");

async function predictClassification(model, image) {
  const tensor = tf.node
    .decodeJpeg(image)
    .resizeNearestNeighbor([224, 224])
    .expandDims()
    .toFloat();

  const prediction = model.predict(tensor);
  const score = await prediction.data();
  const confidenceScore = Math.max(...score) * 100;

  return { confidenceScore, label, explanation, suggestion };
}

module.exports = predictClassification;
