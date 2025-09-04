import { Button } from "@/components/ui/button";
import { Play, Volume2, Subtitles } from "lucide-react";
import aiAvatar from "@/assets/ai-avatar.jpg";

const DemoVideo = () => {
  return (
    <section className="py-20 bg-muted/20">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6">
            Демонстрация AI-репетитора
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Посмотрите, как наш AI-репетитор объясняет сложные темы через жестовый язык
          </p>
        </div>
        
        <div className="max-w-4xl mx-auto">
          {/* Video Player Mockup */}
          <div className="relative bg-card rounded-3xl overflow-hidden shadow-strong animate-fade-in">
            <div className="aspect-video bg-gradient-hero flex items-center justify-center relative">
              {/* Avatar in center */}
              <div className="w-32 h-32 rounded-full overflow-hidden shadow-glow animate-float">
                <img 
                  src={aiAvatar} 
                  alt="AI Tutor demonstrating sign language" 
                  className="w-full h-full object-cover"
                />
              </div>
              
              {/* Play Button Overlay */}
              <div className="absolute inset-0 bg-black/30 flex items-center justify-center group cursor-pointer hover:bg-black/40 transition-colors">
                <Button size="lg" className="w-20 h-20 rounded-full bg-white/20 hover:bg-white/30 group-hover:scale-110 transition-all">
                  <Play className="w-8 h-8 text-white ml-1" />
                </Button>
              </div>
              
              {/* Video Controls Mockup */}
              <div className="absolute bottom-4 left-4 right-4">
                <div className="bg-black/50 rounded-lg p-3 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Button size="sm" variant="ghost" className="text-white hover:text-accent">
                      <Play className="w-4 h-4" />
                    </Button>
                    <Button size="sm" variant="ghost" className="text-white hover:text-accent">
                      <Volume2 className="w-4 h-4" />
                    </Button>
                    <Button size="sm" variant="ghost" className="text-white hover:text-accent">
                      <Subtitles className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="text-white text-sm">
                    2:34 / 8:12
                  </div>
                </div>
              </div>
            </div>
            
            {/* Subtitles Area */}
            <div className="p-6 bg-white border-t">
              <div className="text-center">
                <p className="text-lg text-muted-foreground italic">
                  "Сегодня мы изучаем основы алгебры. Посмотрите на это уравнение..."
                </p>
                <div className="mt-4 flex justify-center gap-2">
                  <span className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm">
                    Русский жестовый язык
                  </span>
                  <span className="px-3 py-1 bg-secondary/10 text-secondary rounded-full text-sm">
                    Субтитры включены
                  </span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Demo Features */}
          <div className="grid md:grid-cols-3 gap-6 mt-12">
            <div className="text-center animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🤟</span>
              </div>
              <h3 className="text-xl font-bold mb-2">Жестовый язык</h3>
              <p className="text-muted-foreground">3D-аватар показывает жесты на казахском и русском языках</p>
            </div>
            
            <div className="text-center animate-fade-in-up" style={{animationDelay: '0.4s'}}>
              <div className="w-16 h-16 bg-gradient-secondary rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">📱</span>
              </div>
              <h3 className="text-xl font-bold mb-2">Адаптивность</h3>
              <p className="text-muted-foreground">Работает на всех устройствах с высоким качеством</p>
            </div>
            
            <div className="text-center animate-fade-in-up" style={{animationDelay: '0.6s'}}>
              <div className="w-16 h-16 bg-accent/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🧠</span>
              </div>
              <h3 className="text-xl font-bold mb-2">AI-помощник</h3>
              <p className="text-muted-foreground">Отвечает на вопросы и адаптируется под уровень студента</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DemoVideo;