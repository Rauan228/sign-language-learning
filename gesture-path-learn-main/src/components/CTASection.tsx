import { Button } from "@/components/ui/button";
import { ArrowRight, Mail, Phone, MapPin } from "lucide-react";

const CTASection = () => {
  return (
    <section className="py-20 bg-gradient-hero relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-10 left-10 w-20 h-20 border-2 border-white rounded-full"></div>
        <div className="absolute top-32 right-20 w-16 h-16 border-2 border-white rounded-full"></div>
        <div className="absolute bottom-20 left-32 w-12 h-12 border-2 border-white rounded-full"></div>
        <div className="absolute bottom-32 right-10 w-24 h-24 border-2 border-white rounded-full"></div>
      </div>
      
      <div className="container mx-auto px-6 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div className="text-center lg:text-left animate-fade-in-up">
            <h2 className="text-4xl lg:text-6xl font-bold text-white mb-6 leading-tight">
              Начните обучение уже{" "}
              <span className="text-accent">сегодня</span>
            </h2>
            
            <p className="text-xl text-white/90 mb-8 leading-relaxed">
              Присоединяйтесь к революции в образовании для людей с ограниченным слухом. 
              Первый урок - бесплатно!
            </p>
            
            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start mb-8">
              <Button size="lg" className="bg-white text-primary hover:bg-white/90 text-lg px-8 py-4 shadow-strong">
                Зарегистрироваться
                <ArrowRight className="ml-2 w-5 h-5" />
              </Button>
              <Button size="lg" variant="outline" className="text-white border-white/30 hover:bg-white/10 text-lg px-8 py-4">
                Связаться с нами
              </Button>
            </div>
            
            {/* Benefits */}
            <div className="flex flex-wrap gap-6 justify-center lg:justify-start text-white/80">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-accent rounded-full"></span>
                <span>Бесплатный пробный период</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-accent rounded-full"></span>
                <span>Сертификаты государственного образца</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-accent rounded-full"></span>
                <span>Поддержка 24/7</span>
              </div>
            </div>
          </div>
          
          {/* Right Content - Contact Info */}
          <div className="animate-fade-in">
            <div className="bg-white/10 backdrop-blur-sm rounded-3xl p-8 shadow-strong">
              <h3 className="text-2xl font-bold text-white mb-6">
                Свяжитесь с нами
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center gap-4 text-white/90">
                  <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center">
                    <Mail className="w-6 h-6" />
                  </div>
                  <div>
                    <div className="font-semibold">Email</div>
                    <div className="text-white/70">info@signlang-edu.kz</div>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 text-white/90">
                  <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center">
                    <Phone className="w-6 h-6" />
                  </div>
                  <div>
                    <div className="font-semibold">Телефон</div>
                    <div className="text-white/70">+7 (727) 123-45-67</div>
                  </div>
                </div>
                
                <div className="flex items-center gap-4 text-white/90">
                  <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center">
                    <MapPin className="w-6 h-6" />
                  </div>
                  <div>
                    <div className="font-semibold">Офис</div>
                    <div className="text-white/70">г. Алматы, ул. Абая 150</div>
                  </div>
                </div>
              </div>
              
              {/* Social Links */}
              <div className="pt-6 mt-6 border-t border-white/20">
                <div className="text-white/70 text-sm mb-3">Следите за нами:</div>
                <div className="flex gap-3">
                  <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center cursor-pointer hover:bg-white/30 transition-colors">
                    <span className="text-lg">📱</span>
                  </div>
                  <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center cursor-pointer hover:bg-white/30 transition-colors">
                    <span className="text-lg">🤟</span>
                  </div>
                  <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center cursor-pointer hover:bg-white/30 transition-colors">
                    <span className="text-lg">📺</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CTASection;