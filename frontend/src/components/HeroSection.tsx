import * as React from 'react';
import { Truck } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { FraudDetectionAnimation } from './FraudDetectionAnimation';
// import exampleImage from 'figma:asset/7623c8d8a0a7a67543f5a0403e31f6b9d0e050aa.png';

interface HeroSectionProps {
  isSignUp: boolean;
}

export function HeroSection({ isSignUp }: HeroSectionProps) {
  return (
    <div className="relative min-h-screen bg-black p-8 flex flex-col items-center justify-center overflow-hidden">
      {/* Geometric background elements */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Large geometric shapes */}
        <div className="absolute top-20 right-20 w-96 h-96 bg-gradient-to-br from-gray-800/20 to-gray-900/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 left-20 w-80 h-80 bg-gradient-to-tr from-gray-700/15 to-gray-800/15 rounded-full blur-3xl"></div>
        
        {/* Angular geometric shapes inspired by Arcana */}
        <div className="absolute top-0 right-0 w-1/3 h-full">
          <div className="absolute top-0 right-0 w-full h-32 bg-gradient-to-l from-gray-800/10 to-transparent transform -skew-y-12"></div>
          <div className="absolute bottom-0 right-0 w-full h-40 bg-gradient-to-l from-gray-900/10 to-transparent transform skew-y-6"></div>
        </div>
        
        {/* Small floating dots */}
        <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-white/60 rounded-full animate-pulse"></div>
        <div className="absolute top-3/4 right-1/3 w-1 h-1 bg-gray-300/40 rounded-full animate-pulse delay-1000"></div>
        <div className="absolute bottom-1/4 left-2/3 w-1.5 h-1.5 bg-white/50 rounded-full animate-pulse delay-500"></div>
      </div>
      
      {/* Large logo/brand element */}
      <div className="relative z-10 mb-8">
        <div className="w-32 h-32 mx-auto relative">
          <div className="absolute inset-0 bg-gradient-to-br from-gray-200 to-gray-400 rounded-3xl transform rotate-12 opacity-10"></div>
          <div className="absolute inset-2 bg-gradient-to-br from-gray-700 to-gray-900 rounded-2xl flex items-center justify-center">
            <svg className="w-16 h-16 text-white" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
          </div>
        </div>
      </div>
      
      {/* Main content */}
      <div className="relative z-10 max-w-lg text-center text-white">
        <h1 className="text-4xl mb-4">
          {isSignUp ? 'Welcome to ShipGuard' : 'Welcome to ShipGuard'}
        </h1>
        
        <p className="text-lg text-gray-300 mb-8 leading-relaxed">
          {isSignUp 
            ? 'ShipGuard helps developers to build organized and well coded dashboards full of beautiful and rich modules. Join us and start building your application today.'
            : 'Monitor and secure your global shipping operations with advanced fraud detection technology powered by AI.'
          }
        </p>
        
        {isSignUp && (
          <p className="text-sm text-gray-400 mb-8">
            More than 17k people joined us, it's your turn
          </p>
        )}
      </div>
      
      {/* Interactive Animation */}
      <div className="relative z-10 w-full max-w-lg mt-8">
        <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/30 mb-6">
          <FraudDetectionAnimation />
        </div>
        
        {/* Feature card */}
        <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl rounded-2xl p-6 border border-gray-700/30">
          <div className="flex items-start space-x-4 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-gray-600 to-gray-800 rounded-xl flex items-center justify-center flex-shrink-0">
              <Truck className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-white mb-2">
                {isSignUp ? 'See Our AI in Action' : 'Real-time Fraud Detection'}
              </h3>
              <p className="text-sm text-gray-300">
                {isSignUp 
                  ? 'Watch how our intelligent system detects and prevents fraudulent shipments in real-time.'
                  : 'AI-powered scanning identifies suspicious patterns and blocks fraudulent activities instantly.'
                }
              </p>
            </div>
          </div>
          
          {/* User avatars or stats */}
          <div className="flex items-center justify-between">
            <div className="flex -space-x-2">
              {[1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="w-8 h-8 bg-gradient-to-br from-gray-600 to-gray-800 rounded-full border-2 border-gray-800 flex items-center justify-center"
                >
                  <span className="text-xs text-white">{i}</span>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-400">
              {isSignUp ? '17k+ users' : '99.9% uptime'}
            </div>
          </div>
        </div>
      </div>
      
      {/* Background image container */}
      <div className="absolute bottom-8 right-8 w-80 h-48 rounded-2xl overflow-hidden opacity-20">
        <ImageWithFallback
          src="https://images.unsplash.com/photo-1651649503984-5b5f3514d6f0?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzaGlwcGluZyUyMGNvbnRhaW5lcnMlMjBsb2dpc3RpY3N8ZW58MXx8fHwxNzU3NTE4ODM3fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
          alt="Shipping Containers"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent"></div>
      </div>
     
      {/* Reference image (small, barely visible) */}
      {/*
      <div className="absolute top-8 left-8 w-20 h-20 rounded-lg overflow-hidden opacity-5">
        <img 
          src={exampleImage} 
          alt="Reference"
          className="w-full h-full object-cover"
        />
      </div>
      */}
    </div>
  );
}
