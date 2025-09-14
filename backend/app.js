// backend/app.js
const express = require('express');
const app = express();
const authRoutes = require('./routes/auth');

app.use(express.json());
app.use('/auth', authRoutes);

app.listen(5000, () => console.log('Backend running on port 5000'));
