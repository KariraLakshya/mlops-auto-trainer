import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { ClerkProvider } from '@clerk/clerk-react'

// REPLACE WITH YOUR KEY FROM CLERK DASHBOARD
const PUBLISHABLE_KEY = "pk_test_ZXBpYy1wYXJyb3QtMjkuY2xlcmsuYWNjb3VudHMuZGV2JA"
if (!PUBLISHABLE_KEY) {
  throw new Error("Missing Publishable Key")
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY}>
      <App />
    </ClerkProvider>
  </React.StrictMode>,
)