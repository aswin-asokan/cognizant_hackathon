// backend/middleware/authMiddleware.js
const { auth } = require('../firebaseAdmin');

const verifyToken = async (req, res, next) => {
  const idToken = req.headers.authorization?.split(' ')[1]; // Bearer <token>
  if (!idToken) return res.status(401).json({ error: 'No token provided' });

  try {
    const decodedToken = await auth.verifyIdToken(idToken);
    req.user = decodedToken;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
};

module.exports = verifyToken;
