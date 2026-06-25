# Vision OMR Mobile Client

React Native mobile application for capturing and submitting OMR sheets.

## Prerequisites

| Tool | Version |
|------|---------|
| Node.js | ≥ 18 |
| React Native CLI | latest |
| Android Studio / Xcode | latest stable |

## Setup

```bash
# Install dependencies
npm install

# iOS: install CocoaPods
cd ios && pod install && cd ..
```

## Running

```bash
# Start Metro bundler
npm start

# Android
npm run android

# iOS
npm run ios
```

## Project Structure

```
src/
├── components/
│   ├── CameraScanner.jsx   # Camera capture + JPEG compression
│   └── ResultsModal.jsx    # Grading results overlay
├── services/
│   └── api.js              # Axios client → data gateway
└── App.jsx                 # Root component + state machine
```

## Environment

Set `DATA_GATEWAY_URL` in your build config (`.env`) to point to the
Node.js data gateway. Defaults to `http://10.0.2.2:3000` (Android emulator).

## Permissions

- **Camera** – required for OMR scanning
- **Internet** – required for API calls
