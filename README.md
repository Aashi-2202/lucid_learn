# LUCID local mentor demo

LUCID is a local, model-agnostic fairness assessment platform for existing binary loan and hiring AI decisions. It evaluates decisions, optional scores, and later outcomes; it does not retrain or alter the model under review.

## Run locally with Docker

1. Start Docker Desktop and confirm its Linux container engine is running.
2. Open PowerShell in this project folder and run:

   ```powershell
   docker compose up --build
   ```

3. Open [http://localhost:3000](http://localhost:3000) for the dashboard.
4. Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

Keep the PowerShell window open while using LUCID. Press `Ctrl+C` there when you want to stop the local services.

## Mentor demo summary

[Download the LUCID mentor demo summary (PDF)](output/pdf/lucid-mentor-demo-summary.pdf).

The initial run seeds two fictional workspaces:

- Northstar Lending with `CreditRisk-v1` and API key `demo-lending-key`.
- TalentBridge Recruiting with `ShortlistAssist-v2` and API key `demo-hiring-key`.

Click **Run assessment** to create the first findings and automatic review cases. **Reset demo data** restores the fictional baseline.

## Run without Docker during development

This is useful while Docker Desktop is unavailable. It uses SQLite only for the local demo; Docker Compose uses PostgreSQL.

1. Create and activate a Python 3.12 virtual environment, then install `services/api/requirements.txt`.
2. Start the API from `services/api` with `python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`.
3. In `apps/web`, run `pnpm install`, then `pnpm dev`.
4. Open the dashboard at [http://127.0.0.1:3000](http://127.0.0.1:3000). Both services are bound to loopback only, so they are not exposed to other devices on your network.

## Demo connector

After the API is running, use `samples/send_demo_event.py` to send a fictional decision and outcome. The default key is the seeded lending model key.

## Important boundaries

- The demo accepts only pseudonymous audit data. It rejects obvious direct-identifier columns and fields.
- Historical CSV imports are optional. The API can gather evidence from new decisions over time.
- Assessments run manually in this version; no scheduled jobs are configured.
- Findings are labelled `needs_review`, `within_policy`, or `inconclusive`; LUCID does not issue legal conclusions.
