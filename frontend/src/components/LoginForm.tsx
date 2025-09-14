import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Separator } from './ui/separator';
import { Checkbox } from './ui/checkbox';
import { Eye, EyeOff, Mail, Lock, User, Building2, Github } from 'lucide-react';

interface LoginFormProps {
  isSignUp: boolean;
  onToggleMode: () => void;
}

export function LoginForm({ isSignUp, onToggleMode }: LoginFormProps) {
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    company: ''
  });

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isSignUp) {
      console.log('Sign up attempt:', formData);
    } else {
      console.log('Login attempt:', { email: formData.email, password: formData.password });
    }
  };

  return (
    <div className="w-full max-w-md">
      {/* Logo */}
      <div className="mb-8">
        <div className="w-16 h-16 bg-black rounded-2xl flex items-center justify-center mb-6 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-black"></div>
          <div className="relative">
            <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
          </div>
        </div>
        <h1 className="text-3xl mb-2 text-black">
          {isSignUp ? 'Create Account' : 'Sign In'}
        </h1>
        <p className="text-gray-600">
          {isSignUp 
            ? 'Join our fraud detection platform'
            : 'Please sign in to your account'
          }
        </p>
      </div>
      
      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {isSignUp && (
          <>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="firstName" className="text-sm text-gray-700">First Name</Label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                  <Input
                    id="firstName"
                    placeholder="John"
                    value={formData.firstName}
                    onChange={(e) => handleInputChange('firstName', e.target.value)}
                    className="pl-10 h-12 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-black focus:border-black transition-all"
                    required
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="lastName" className="text-sm text-gray-700">Last Name</Label>
                <Input
                  id="lastName"
                  placeholder="Doe"
                  value={formData.lastName}
                  onChange={(e) => handleInputChange('lastName', e.target.value)}
                  className="h-12 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-black focus:border-black transition-all"
                  required
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="company" className="text-sm text-gray-700">Company</Label>
              <div className="relative">
                <Building2 className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  id="company"
                  placeholder="Acme Corp"
                  value={formData.company}
                  onChange={(e) => handleInputChange('company', e.target.value)}
                  className="pl-10 h-12 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-black focus:border-black transition-all"
                  required
                />
              </div>
            </div>
          </>
        )}
        
        <div className="space-y-2">
          <Label htmlFor="email" className="text-sm text-gray-700">Email Address</Label>
          <div className="relative">
            <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              id="email"
              type="email"
              placeholder="john@example.com"
              value={formData.email}
              onChange={(e) => handleInputChange('email', e.target.value)}
              className="pl-10 h-12 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-black focus:border-black transition-all"
              required
            />
          </div>
        </div>
        
        <div className="space-y-2">
          <Label htmlFor="password" className="text-sm text-gray-700">Password</Label>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              id="password"
              type={showPassword ? 'text' : 'password'}
              placeholder="••••••••"
              value={formData.password}
              onChange={(e) => handleInputChange('password', e.target.value)}
              className="pl-10 pr-10 h-12 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-black focus:border-black transition-all"
              required
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
            >
              {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
        </div>
        
        {isSignUp && (
          <div className="space-y-2">
            <Label htmlFor="confirmPassword" className="text-sm text-gray-700">Confirm Password</Label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                id="confirmPassword"
                type="password"
                placeholder="••••••••"
                value={formData.confirmPassword}
                onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                className="pl-10 h-12 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-black focus:border-black transition-all"
                required
              />
            </div>
          </div>
        )}
        
        {!isSignUp && (
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Checkbox 
                id="remember" 
                checked={rememberMe}
                onCheckedChange={setRememberMe}
                className="border-gray-300"
              />
              <Label htmlFor="remember" className="text-sm text-gray-600">Remember me</Label>
            </div>
            <button
              type="button"
              className="text-sm text-gray-600 hover:text-black transition-colors"
            >
              Forgot Password
            </button>
          </div>
        )}
        
        <Button type="submit" className="w-full h-12 bg-black hover:bg-gray-800 text-white rounded-lg transition-all duration-200">
          {isSignUp ? 'Create Account' : 'Sign In'}
        </Button>
      </form>
      
      <div className="relative my-8">
        <div className="absolute inset-0 flex items-center">
          <Separator className="w-full border-gray-200" />
        </div>
        <div className="relative flex justify-center text-xs uppercase">
          <span className="bg-white px-4 text-gray-500">Or continue with</span>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-3">
        <Button variant="outline" className="h-12 border-gray-200 hover:bg-gray-50 transition-all" type="button">
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
            <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
            <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
          </svg>
        </Button>
        <Button variant="outline" className="h-12 border-gray-200 hover:bg-gray-50 transition-all" type="button">
          <Github className="w-5 h-5" />
        </Button>
        <Button variant="outline" className="h-12 border-gray-200 hover:bg-gray-50 transition-all" type="button">
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
            <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
          </svg>
        </Button>
      </div>
      
      <div className="text-center mt-8">
        <span className="text-sm text-gray-600">
          {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
        </span>
        <button
          type="button"
          onClick={onToggleMode}
          className="text-sm text-black hover:underline transition-all"
        >
          {isSignUp ? 'Sign up' : 'Sign up'}
        </button>
      </div>
    </div>
  );
}