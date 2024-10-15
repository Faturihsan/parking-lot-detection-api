const express = require('express');
const bodyParser = require('body-parser');

// const authRoutes = require('./api/auth/auth.route');
const predictRoutes = require('./api/predict/predict.route');

const { loadObjectDetectionModel, loadRegressionModel } = require("./utils/loadModel");
// const verifyToken = require('./middlewares/auth.middleware');
// const apiLimiter = require('./middlewares/ratelimit.middleware');
const app = express();




const port = parseInt(process.env.PORT) || process.argv[3] || 3000;

app.use(bodyParser.json());



(async () => {
  try {
    const objectDetectionModel = await loadObjectDetectionModel();
    // const regressionModel = await loadRegressionModel();

    app.use((req, res, next) => {
      req.models = {
        objectDetectionModel,
      };
      next();
    });

    // app.use('/api/auth', apiLimiter.authLimiter, authRoutes);
    app.use('/api/object-detection', predictRoutes);

    app.listen(port, () => {
      console.log(`Listening on http://localhost:${port}`);
    });
  } catch (error) {
    console.error('Failed to load models:', error);
  }
})();