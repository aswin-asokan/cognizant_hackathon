import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyAulJVe859Pjj3ECwX6lAQods58LM50WY4",
  authDomain: "fraud-shipment.firebaseapp.com",
  projectId: "fraud-shipment",
  storageBucket: "fraud-shipment.firebasestorage.app",
  messagingSenderId: "967660476984",
  appId: "1:967660476984:web:535f7dc2635c7cb375a4cd",
  measurementId: "G-FQ86M08N7Y"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

export { app, analytics };
