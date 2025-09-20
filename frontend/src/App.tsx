// App.tsx
import { useState } from 'react';
import { LoginForm } from './components/LoginForm';
import { HeroSection } from './components/HeroSection';
import Dashboard from './components/Dashboard'; // Import your new Dashboard component

export default function App() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false); // New state to track login status

  const toggleMode = () => {
    setIsSignUp(!isSignUp);
  };

  const handleAuthSuccess = () => {
    setIsLoggedIn(true); // Navigate to dashboard by setting the state
  };

  return (
    <>
      {isLoggedIn ? (
        // Render the Dashboard component if the user is logged in
        <Dashboard />
      ) : (
        // Otherwise, show the login/signup page
        <div className="min-h-screen flex bg-gray-50">
          {/* Left side - Login Form */}
          <div className="flex-1 flex items-center justify-center p-8 lg:p-12">
            <div className="w-full max-w-md">
              <LoginForm 
                isSignUp={isSignUp} 
                onToggleMode={toggleMode} 
                onAuthSuccess={handleAuthSuccess} // Pass the new callback function
              />
            </div>
          </div>
          
          {/* Right side - Hero Section */}
          <div className="hidden lg:block flex-1 relative">
            <HeroSection isSignUp={isSignUp} />
          </div>
        </div>
      )}
    </>
  );
}