import { Button } from "@/components/ui/button";
import { Play, Users, BookOpen } from "lucide-react";
import { Link } from "react-router-dom";
import heroImage from "@/assets/hero-education.jpg";
import aiAvatar from "@/assets/ai-avatar.jpg";

const HeroSection = () => {
  return (
    <section className="relative min-h-screen flex items-center bg-gradient-hero overflow-hidden">
      {/* Background Image */}
      <div className="absolute inset-0 z-0">
        <img 
          src={heroImage} 
          alt="Educational platform hero" 
          className="w-full h-full object-cover opacity-20"
        />
        <div className="absolute inset-0 bg-gradient-hero opacity-80"></div>
      </div>
      
      <div className="container mx-auto px-6 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div className="text-center lg:text-left animate-fade-in-up">
            <h1 className="text-5xl lg:text-7xl font-bold text-white mb-6 leading-tight">
              Образование через{" "}
              <span className="text-gradient-primary">жестовый язык</span>
            </h1>
            
            <p className="text-xl lg:text-2xl text-white/90 mb-8 font-light leading-relaxed">
              AI-репетитор с поддержкой казахского и русского жестового языка. 
              Интерактивное обучение с 3D-аватарами и визуальными материалами.
            </p>
            
            {/* Stats */}
            <div className="flex flex-wrap gap-8 justify-center lg:justify-start mb-10">
              <div className="flex items-center gap-2 text-white/80">
                <Users className="w-6 h-6 text-accent" />
                <span className="text-lg font-semibold">10,000+ студентов</span>
              </div>
              <div className="flex items-center gap-2 text-white/80">
                <BookOpen className="w-6 h-6 text-accent" />
                <span className="text-lg font-semibold">50+ курсов</span>
              </div>
            </div>
            
            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Button asChild size="lg" className="btn-hero text-lg px-8 py-4">
                <Link to="/courses">
                  Начать обучение
                  <Play className="ml-2 w-5 h-5" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="text-white border-white/30 hover:bg-white/10 text-lg px-8 py-4">
                <Link to="/ai-assistant">
                  Демо AI-репетитора
                </Link>
              </Button>
            </div>
          </div>
          
          {/* Right Content - Avatar */}
          <div className="flex justify-center lg:justify-end animate-fade-in">
            <div className="relative">
              <div className="w-80 h-80 lg:w-96 lg:h-96 rounded-full overflow-hidden shadow-glow animate-float">
                <img 
                  src={aiAvatar} 
                  alt="AI Tutor Avatar" 
                  className="w-full h-full object-cover"
                />
              </div>
              
              {/* Floating Elements */}
              <div className="absolute -top-4 -right-4 w-16 h-16 bg-accent rounded-full flex items-center justify-center shadow-strong animate-pulse-glow">
                <span className="text-2xl">🤟</span>
              </div>
              
              <div className="absolute -bottom-4 -left-4 w-12 h-12 bg-secondary rounded-full flex items-center justify-center shadow-strong animate-pulse-glow" style={{animationDelay: '1s'}}>
                <span className="text-lg">📚</span>
              </div>
              
              <div className="absolute top-1/2 -left-8 w-14 h-14 bg-primary rounded-full flex items-center justify-center shadow-strong animate-pulse-glow" style={{animationDelay: '2s'}}>
                <span className="text-xl">🧠</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-float">
        <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center">
          <div className="w-1 h-3 bg-white/60 rounded-full mt-2 animate-bounce"></div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;