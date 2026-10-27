---
name: stripe-payments
description: "Stripe payment and payout management: check account balances, track bank payouts, generate payment links for services, and audit incoming revenue."
version: 1.0.0
author: Hermes Autonomous Evolution Engine
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Finance, Stripe, Payments, Payouts, Billing]
---

# Stripe Payments & Payouts Skill

## Overview
Programmatic Stripe integration for inspecting live balances, verifying bank payout schedules, generating instant payment links, and monitoring digital revenue.

## CLI Commands
Available globally via `stripe-tool`:
```bash
stripe-tool status    # Check Stripe credentials and account health
stripe-tool balance   # Inspect available and pending balances
stripe-tool payouts   # View recent transfers to your bank account
stripe-tool link --amount 25.00 --currency usd --title "AI Code Audit"  # Generate instant Stripe payment link
```

## Setup
Set your Stripe API keys in `~/.hermes/.env`:
```env
STRIPE_SECRET_KEY=sk_live_... (or sk_test_...)
STRIPE_PUBLISHABLE_KEY=pk_live_...
```
Keys can be obtained from: https://dashboard.stripe.com/apikeys
