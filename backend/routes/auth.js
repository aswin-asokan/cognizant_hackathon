// backend/routes/auth.js
const express = require('express');
const router = express.Router();
const { auth, db } = require('../firebaseAdmin');

// ----------------------
// Email/Password Signup
// ----------------------
router.post('/signup', async (req, res) => {
  const { firstName, lastName, company, email, password } = req.body;
  try {
    // Create user in Firebase Auth
    const userRecord = await auth.createUser({
      email,
      password,
      displayName: `${firstName} ${lastName}`,
    });

    // Store additional info in Firestore
    await db.collection('users').doc(userRecord.uid).set({
      firstName,
      lastName,
      company,
      email,
      provider: 'password',
      createdAt: new Date(),
    });

    res.status(200).json({ message: 'User created', uid: userRecord.uid });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// ----------------------
// Social Login
// ----------------------
router.post('/social-login', async (req, res) => {
  const { idToken } = req.body; // Firebase ID token from frontend
  try {
    const decodedToken = await auth.verifyIdToken(idToken);
    const uid = decodedToken.uid;

    // Store user in Firestore if first login
    const userDoc = db.collection('users').doc(uid);
    const doc = await userDoc.get();
    if (!doc.exists) {
      await userDoc.set({
        firstName: decodedToken.name?.split(' ')[0] || '',
        lastName: decodedToken.name?.split(' ')[1] || '',
        email: decodedToken.email,
        provider: decodedToken.firebase.sign_in_provider,
        createdAt: new Date(),
      });
    }

    res.json({ message: 'Login successful', uid });
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
});

module.exports = router;
