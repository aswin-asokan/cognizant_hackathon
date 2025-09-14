import { useState } from 'react';
import { LoginForm } from './components/LoginForm';
import { HeroSection } from './components/HeroSection';

export default function App() {
  const [isSignUp, setIsSignUp] = useState(false);

  const toggleMode = () => {
    setIsSignUp(!isSignUp);
  };

  return (
    <div className="min-h-screen flex bg-gray-50">
      {/* Left side - Login Form */}
      <div className="flex-1 flex items-center justify-center p-8 lg:p-12">
        <div className="w-full max-w-md">
          <LoginForm isSignUp={isSignUp} onToggleMode={toggleMode} />
        </div>
      </div>
      
      {/* Right side - Hero Section */}
      <div className="hidden lg:block flex-1 relative">
        <HeroSection isSignUp={isSignUp} />
      </div>
    </div>
  );
}