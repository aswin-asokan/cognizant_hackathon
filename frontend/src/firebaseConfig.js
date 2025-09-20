import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getStorage } from "firebase/storage"; // <-- Add this import


const firebaseConfig = {
  apiKey: "AIzaSyAulJVe859Pjj3ECwX6lAQods58LM50WY4",
  authDomain: "fraud-shipment.firebaseapp.com",
  projectId: "fraud-shipment",
  storageBucket: "fraud-shipment.appspot.com",
  messagingSenderId: "967660476984",
  appId: "1:967660476984:web:535f7dc2635c7cb375a4cd",
  measurementId: "G-FQ86M08N7Y"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const storage = getStorage(app); // <-- Add this line to initialize storage


export { app, analytics, storage }; // <-- Add 'storage' to the export list