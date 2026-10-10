// EpiRadar SMS Gateway - Twilio Version
// Run: node backend/sms_server.js

const express = require('express');
const twilio = require('twilio');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

// ── Twilio Credentials ────────────────────────────────────────
const TWILIO_SID   = process.env.TWILIO_ACCOUNT_SID || "YOUR_TWILIO_ACCOUNT_SID";
const TWILIO_TOKEN = process.env.TWILIO_AUTH_TOKEN  || "YOUR_TWILIO_AUTH_TOKEN";
const TWILIO_FROM  = process.env.TWILIO_PHONE_NUMBER || "+10000000000";

// ── Agent Store ───────────────────────────────────────────────
let agents = [];

// ── Health Check ──────────────────────────────────────────────
app.get('/health', (req, res) => {
    res.json({ status: 'ok', service: 'EpiRadar SMS Gateway', provider: 'Twilio' });
});

// ── Register Agent ────────────────────────────────────────────
app.post('/register-agent', (req, res) => {
    const { username, phone, district } = req.body;
    if (!username || !phone || !district)
        return res.status(400).json({ success: false, error: "Missing fields" });
    agents = agents.filter(a => a.phone !== phone);
    agents.push({ username, phone, district });
    console.log(`[REG] ✅ ${username} | ${district} | ${phone}`);
    res.json({ success: true });
});

// ── Get Agents ────────────────────────────────────────────────
app.get('/agents', (req, res) => {
    res.json({ agents, total: agents.length });
});

// ── Send Single SMS ───────────────────────────────────────────
app.post('/send-sms', async (req, res) => {
    const { to, message } = req.body;
    if (!to || !message)
        return res.status(400).json({ success: false, error: "Missing fields" });

    const toNumber = '+91' + to.toString().replace(/^\+91|^91/, '').trim();
    try {
        const client = twilio(TWILIO_SID, TWILIO_TOKEN);
        const msg = await client.messages.create({ body: message, from: TWILIO_FROM, to: toNumber });
        console.log(`[OK] ✅ SMS → ${toNumber} | SID: ${msg.sid}`);
        res.json({ success: true, sid: msg.sid });
    } catch (e) {
        console.error(`[ERR] ❌ ${e.message}`);
        res.status(500).json({ success: false, error: e.message });
    }
});

// ── District Alert ────────────────────────────────────────────
app.post('/send-district-alert', async (req, res) => {
    const { district, message } = req.body;
    if (!district)
        return res.status(400).json({ success: false, error: "Missing district" });

    const targets = agents.filter(a => a.district === district);
    if (!targets.length)
        return res.status(404).json({ success: false, error: `No agents in ${district}` });

    const alertMsg = message ||
`🚨 HIGH RISK ALERT!
District: ${district}
Immediate action required!
EpiRadar - TN Health
Login: localhost:8501
Helpline: 104`;

    console.log(`[ALERT] 🚨 Sending to ${targets.length} agents in ${district}`);

    const client = twilio(TWILIO_SID, TWILIO_TOKEN);
    const results = [];

    for (const agent of targets) {
        const toNumber = '+91' + agent.phone.toString().replace(/^\+91|^91/, '').trim();
        try {
            const msg = await client.messages.create({
                body: `Dear ${agent.username},\n${alertMsg}`,
                from: TWILIO_FROM,
                to: toNumber
            });
            results.push({ agent: agent.username, phone: agent.phone, success: true });
            console.log(`[OK] ✅ ${agent.username} (${toNumber})`);
        } catch (e) {
            results.push({ agent: agent.username, phone: agent.phone, success: false, error: e.message });
            console.error(`[FAIL] ❌ ${agent.username}: ${e.message}`);
        }
        await new Promise(r => setTimeout(r, 300));
    }

    const sent = results.filter(r => r.success).length;
    res.json({ success: true, sent, total: targets.length, results });
});

// ── Start ─────────────────────────────────────────────────────
const PORT = 3001;
app.listen(PORT, '0.0.0.0', () => {
    console.log('=========================================');
    console.log('  🏥 EpiRadar SMS Gateway Active');
    console.log('  📱 Provider: Twilio');
    console.log(`  🔗 Running on: http://localhost:${PORT}`);
    console.log('=========================================');
});
