import React from 'react';
import { FaCheckCircle, FaTimesCircle, FaSpinner, FaClock, FaForward } from 'react-icons/fa';
// Note: We don't need inline styles anymore because App.css handles it!

const Sidebar = ({ steps, runUrl, isOpen, status }) => {
  return (
    <div className={`sidebar ${isOpen ? 'open' : ''}`}>
      <h2>🚀 Mission Control</h2>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        {steps.map((step, idx) => {
          let icon = <FaClock color="#475569" />; // Gray
          let textClass = "step-text";

          if (step.status === 'in_progress') {
            icon = <FaSpinner className="icon-spin" color="#3b82f6" />; // Blue Spinner
            textClass = "step-text active";
          } else if (step.status === 'completed') {
            if (step.conclusion === 'failure') {
              icon = <FaTimesCircle color="#ef4444" />; // Red
            } else if (step.conclusion === 'skipped') {
              icon = <FaForward color="#64748b" />;
            } else {
              icon = <FaCheckCircle color="#10b981" />; // Green
              textClass = "step-text active";
            }
          }

          // Staggered animation delay for cool effect
          return (
            <div key={idx} className="step-item" style={{ animationDelay: `${idx * 0.1}s` }}>
              <div className="step-icon">{icon}</div>
              <div className={textClass}>
                {step.name}
              </div>
            </div>
          );
        })}
        
        {steps.length === 0 && (
            <p style={{color: '#64748b', fontStyle: 'italic', marginTop: '20px'}}>
                System Standby...
            </p>
        )}
      </div>

      {status === "completed" && (
         <div style={{ 
             padding: '12px', background: 'rgba(16, 185, 129, 0.1)', 
             color: '#10b981', borderRadius: '8px', 
             textAlign: 'center', marginBottom: '15px', border: '1px solid rgba(16, 185, 129, 0.2)' 
         }}>
           ✨ Pipeline Successful
         </div>
      )}

      {runUrl && (
        <a href={runUrl} target="_blank" rel="noreferrer" className="logs-btn">
          View Raw Logs on GitHub
        </a>
      )}
    </div>
  );
};

export default Sidebar;