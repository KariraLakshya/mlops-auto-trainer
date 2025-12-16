require('dotenv').config();
const express = require('express');
const multer = require('multer');
const { Octokit } = require('octokit');
const cors = require('cors');
const { ClerkExpressRequireAuth } = require('@clerk/clerk-sdk-node'); // Middleware
const clerk = require('@clerk/clerk-sdk-node'); // Main SDK

const app = express();
const port = process.env.PORT || 3000;

// 1. Middleware Setup
app.use(cors()); // Allow requests from Frontend
app.use(express.json());

// 2. File Upload Config (Max 95MB)
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 95 * 1024 * 1024 }
});

// --- HELPER: Get GitHub Token from Clerk ---
// This communicates with Clerk to get the OAuth token for the logged-in user
const getGithubTokenFromClerk = async (userId) => {
    try {
        // We request the token associated with the 'oauth_github' provider
        const response = await clerk.users.getUserOauthAccessToken(userId, 'oauth_github');
        
        if (response.length === 0 || !response[0].token) {
            throw new Error("User has not authenticated with GitHub via Clerk.");
        }
        
        return response[0].token;
    } catch (err) {
        console.error("❌ Clerk OAuth Error:", err);
        throw new Error("Failed to retrieve GitHub token. Please Sign In again.");
    }
};

// --- ROUTE 1: TRIGGER TRAINING (Protected) ---
// ClerkExpressRequireAuth() ensures only logged-in users can hit this
app.post('/api/trigger', ClerkExpressRequireAuth(), upload.single('file'), async (req, res) => {
    try {
        // A. Get User ID & Token
        const userId = req.auth.userId;
        const githubToken = await getGithubTokenFromClerk(userId);
        
        // B. Initialize Octokit as the USER
        const userOctokit = new Octokit({ auth: githubToken });

        const { owner, repo } = req.body;
        const file = req.file;

        if (!file || !owner || !repo) {
            return res.status(400).json({ error: "Missing file, owner, or repo." });
        }

        console.log(`🚀 User ${userId} triggering pipeline for ${owner}/${repo}...`);

        // C. Upload Data to GitHub
        const contentEncoded = file.buffer.toString('base64');
        const message = `data: Upload via Clerk Auth Dashboard`;
        const path = "data/data.csv"; // Fixed path expected by manifest

        // Check if file exists to get SHA (needed for updates)
        let sha;
        try {
            const { data: existing } = await userOctokit.request('GET /repos/{owner}/{repo}/contents/{path}', {
                owner, repo, path
            });
            sha = existing.sha;
        } catch (e) { /* File doesn't exist, proceed to create */ }

        // Perform Upload
        await userOctokit.request('PUT /repos/{owner}/{repo}/contents/{path}', {
            owner, repo, path,
            message,
            content: contentEncoded,
            sha
        });

        // D. Dispatch Workflow
        await userOctokit.request('POST /repos/{owner}/{repo}/actions/workflows/retrain_v3.yml/dispatches', {
            owner, repo,
            ref: 'main'
        });

        // E. Wait & Fetch Run ID
        // We wait 2s to allow GitHub to register the event
        await new Promise(r => setTimeout(r, 2000));
        
        const { data: runs } = await userOctokit.request('GET /repos/{owner}/{repo}/actions/runs', {
            owner, repo,
            per_page: 1
        });

        const runId = runs.workflow_runs[0]?.id;
        console.log(`✅ Triggered Run ID: ${runId}`);

        res.json({ success: true, runId });

    } catch (error) {
        console.error("❌ Backend Trigger Error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// --- ROUTE 2: POLL STATUS (Protected) ---
app.get('/api/status/:runId', ClerkExpressRequireAuth(), async (req, res) => {
    try {
        const userId = req.auth.userId;
        const githubToken = await getGithubTokenFromClerk(userId);
        const userOctokit = new Octokit({ auth: githubToken });

        const { runId } = req.params;
        const { owner, repo } = req.query;

        // Fetch Run Details
        const { data: run } = await userOctokit.request('GET /repos/{owner}/{repo}/actions/runs/{run_id}', {
            owner, repo, run_id: runId
        });

        // Fetch Jobs (Steps)
        const { data: jobs } = await userOctokit.request('GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs', {
            owner, repo, run_id: runId
        });

        res.json({
            status: run.status,
            conclusion: run.conclusion,
            html_url: run.html_url,
            steps: jobs.jobs[0]?.steps || []
        });

    } catch (error) {
        console.error("❌ Polling Error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// --- ERROR HANDLING ---
app.use((err, req, res, next) => {
    if (err.message === 'Unauthenticated') {
        return res.status(401).json({ error: "Invalid Clerk Session" });
    }
    res.status(500).json({ error: err.message });
});

// --- START SERVER ---
app.listen(port, () => {
    console.log(`📡 Clerk-Backend listening at http://localhost:${port}`);
});