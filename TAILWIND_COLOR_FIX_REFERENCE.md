# Tailwind Color Fix Reference

## Current Issue
Build fails with missing Tailwind color variants. App.css uses colors that don't exist in tailwind.config.js.

## Complete Color Scales Needed

Add these to `frontend/tailwind.config.js` in the `colors` section:

```javascript
primary: {
  50: '#eff6ff',
  100: '#dbeafe', 
  200: '#bfdbfe',
  300: '#93c5fd',
  400: '#60a5fa',
  500: '#3b82f6',  // ✅ exists
  600: '#2563eb',  // ✅ exists  
  700: '#1d4ed8',  // ✅ exists
  800: '#1e40af',
  900: '#1e3a8a',
},
success: {
  50: '#ecfdf5',
  100: '#d1fae5',
  200: '#a7f3d0', 
  300: '#6ee7b7',
  400: '#34d399',
  500: '#10b981',  // ✅ exists
  600: '#059669',  // ✅ exists
  700: '#047857',  // ✅ exists  
  800: '#065f46',
  900: '#064e3b',
},
warning: {
  50: '#fffbeb',
  100: '#fef3c7',
  200: '#fde68a',
  300: '#fcd34d', 
  400: '#fbbf24',
  500: '#f59e0b',  // ✅ exists
  600: '#d97706',  // ✅ exists
  700: '#b45309',
  800: '#92400e', 
  900: '#78350f',
},
error: {
  50: '#fef2f2',
  100: '#fee2e2',
  200: '#fecaca',
  300: '#fca5a5',
  400: '#f87171',
  500: '#ef4444',  // ✅ exists
  600: '#dc2626',  // ✅ exists
  700: '#b91c1c',  // ✅ exists
  800: '#991b1b',
  900: '#7f1d1d',
},
```

## How to Apply Fix

1. Open `frontend/tailwind.config.js`
2. Replace the incomplete color definitions with complete scales above
3. Run `cd frontend && npm run build`
4. Should build successfully after this fix

## Current Build Error Pattern
```
The `bg-error-50` class does not exist
The `bg-warning-100` class does not exist  
The `bg-success-200` class does not exist
```

These errors indicate the CSS is trying to use color variants that aren't defined in the Tailwind config.