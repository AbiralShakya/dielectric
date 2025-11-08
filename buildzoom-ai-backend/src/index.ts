import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { remodelRouter } from './routes/remodel';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' })); // Allow large images

// Routes
app.use('/api', remodelRouter);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`🚀 BuildZoom AI Backend running on port ${PORT}`);
});
