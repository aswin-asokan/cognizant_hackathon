const admin = require('firebase-admin');
const serviceAccount = require('./serviceAccountKey.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  storageBucket: "fraud-shipment.appspot.com", // optional if using Storage
});

const db = admin.firestore();
const auth = admin.auth();
const bucket = admin.storage().bucket(); // optional

module.exports = { admin, db, auth, bucket };
