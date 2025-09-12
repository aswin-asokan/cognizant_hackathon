import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Truck, Shield, AlertTriangle, CheckCircle, Zap, Star } from 'lucide-react';

export function FraudDetectionAnimation() {
  const [animationState, setAnimationState] = useState<'idle' | 'scanning' | 'detected' | 'celebrating'>('idle');
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; delay: number }>>([]);

  useEffect(() => {
    const runAnimation = async () => {
      // Wait a bit then start the sequence
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAnimationState('scanning');
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setAnimationState('detected');
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setAnimationState('celebrating');
      
      // Generate celebration particles
      const newParticles = Array.from({ length: 12 }, (_, i) => ({
        id: i,
        x: Math.random() * 300,
        y: Math.random() * 200,
        delay: Math.random() * 0.5
      }));
      setParticles(newParticles);
      
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      setAnimationState('idle');
      setParticles([]);
    };

    const interval = setInterval(runAnimation, 8000);
    runAnimation(); // Run immediately

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative w-80 h-60 mx-auto">
      {/* Background scanning grid */}
      <motion.div 
        className="absolute inset-0 opacity-20"
        animate={{
          opacity: animationState === 'scanning' ? 0.4 : 0.1
        }}
        transition={{ duration: 0.5 }}
      >
        <svg className="w-full h-full" viewBox="0 0 320 240">
          <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" className="text-white/30" />
        </svg>
      </motion.div>

      {/* Truck container */}
      <motion.div
        className="absolute top-20 left-0"
        animate={{
          x: animationState === 'idle' ? -100 : animationState === 'celebrating' ? 100 : 80
        }}
        transition={{
          duration: animationState === 'idle' ? 2 : 1.5,
          ease: "easeInOut"
        }}
      >
        <div className="relative">
          {/* Truck */}
          <motion.div 
            className="w-20 h-12 bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg relative shadow-lg"
            animate={{
              scale: animationState === 'detected' ? [1, 1.1, 1] : 1,
              rotate: animationState === 'detected' ? [0, -2, 2, 0] : 0
            }}
            transition={{ duration: 0.5, repeat: animationState === 'detected' ? 2 : 0 }}
          >
            <Truck className="absolute inset-0 m-auto w-8 h-8 text-white" />
            
            {/* Suspicious indicator */}
            <AnimatePresence>
              {animationState === 'detected' && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  exit={{ scale: 0 }}
                  className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center"
                >
                  <AlertTriangle className="w-3 h-3 text-white" />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Truck wheels */}
          <div className="absolute -bottom-2 left-1 w-4 h-4 bg-gray-800 rounded-full"></div>
          <div className="absolute -bottom-2 right-1 w-4 h-4 bg-gray-800 rounded-full"></div>
        </div>
      </motion.div>

      {/* Scanning beam */}
      <AnimatePresence>
        {animationState === 'scanning' && (
          <motion.div
            initial={{ x: -50, opacity: 0 }}
            animate={{ x: 300, opacity: [0, 1, 1, 0] }}
            exit={{ opacity: 0 }}
            transition={{ duration: 2, ease: "linear" }}
            className="absolute top-0 w-1 h-full bg-gradient-to-b from-green-400 via-green-500 to-transparent"
          />
        )}
      </AnimatePresence>

      {/* Detection shield */}
      <motion.div
        className="absolute top-12 right-20"
        animate={{
          scale: animationState === 'detected' || animationState === 'celebrating' ? 1 : 0,
          rotate: animationState === 'celebrating' ? 360 : 0
        }}
        transition={{ duration: 0.5 }}
      >
        <div className="relative">
          <motion.div
            className="w-16 h-16 bg-gradient-to-br from-green-400 to-green-600 rounded-full flex items-center justify-center shadow-lg"
            animate={{
              boxShadow: animationState === 'celebrating' 
                ? ["0 0 0 0 rgba(34, 197, 94, 0.7)", "0 0 0 20px rgba(34, 197, 94, 0)", "0 0 0 0 rgba(34, 197, 94, 0)"]
                : "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
            }}
            transition={{ duration: 1, repeat: animationState === 'celebrating' ? Infinity : 0 }}
          >
            <Shield className="w-8 h-8 text-white" />
          </motion.div>
          
          <AnimatePresence>
            {animationState === 'celebrating' && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                className="absolute -top-1 -right-1 w-6 h-6 bg-yellow-400 rounded-full flex items-center justify-center"
              >
                <CheckCircle className="w-4 h-4 text-yellow-800" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* Status text */}
      <motion.div
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        animate={{ opacity: animationState === 'idle' ? 0 : 1 }}
        transition={{ duration: 0.3 }}
      >
        <motion.div
          className="bg-black/80 backdrop-blur-sm rounded-full px-4 py-2 text-sm text-white"
          animate={{
            backgroundColor: 
              animationState === 'detected' ? 'rgba(239, 68, 68, 0.9)' :
              animationState === 'celebrating' ? 'rgba(34, 197, 94, 0.9)' :
              'rgba(0, 0, 0, 0.8)'
          }}
        >
          {animationState === 'scanning' && (
            <span className="flex items-center gap-2">
              <Zap className="w-4 h-4 animate-pulse" />
              Scanning shipment...
            </span>
          )}
          {animationState === 'detected' && (
            <span className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Fraud detected!
            </span>
          )}
          {animationState === 'celebrating' && (
            <span className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4" />
              Threat blocked! 🎉
            </span>
          )}
        </motion.div>
      </motion.div>

      {/* Celebration particles */}
      <AnimatePresence>
        {animationState === 'celebrating' && particles.map((particle) => (
          <motion.div
            key={particle.id}
            initial={{ 
              x: 160, 
              y: 120, 
              scale: 0,
              rotate: 0 
            }}
            animate={{ 
              x: particle.x, 
              y: particle.y, 
              scale: [0, 1, 0],
              rotate: 360 
            }}
            transition={{ 
              duration: 2,
              delay: particle.delay,
              ease: "easeOut"
            }}
            className="absolute"
          >
            <Star className="w-4 h-4 text-yellow-400 fill-current" />
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Floating success icons */}
      <AnimatePresence>
        {animationState === 'celebrating' && (
          <>
            <motion.div
              initial={{ y: 100, opacity: 0 }}
              animate={{ y: -20, opacity: [0, 1, 0] }}
              transition={{ duration: 2, delay: 0.5 }}
              className="absolute left-10 top-10"
            >
              <CheckCircle className="w-8 h-8 text-green-400" />
            </motion.div>
            <motion.div
              initial={{ y: 100, opacity: 0 }}
              animate={{ y: -20, opacity: [0, 1, 0] }}
              transition={{ duration: 2, delay: 0.8 }}
              className="absolute right-10 top-16"
            >
              <Shield className="w-6 h-6 text-blue-400" />
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Scanning radar effect */}
      <AnimatePresence>
        {animationState === 'scanning' && (
          <motion.div
            initial={{ scale: 0, opacity: 0.8 }}
            animate={{ scale: 2, opacity: 0 }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-20 h-20 border-2 border-green-400 rounded-full"
          />
        )}
      </AnimatePresence>
    </div>
  );
}