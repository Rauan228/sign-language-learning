import { Button } from "@/components/ui/button";
import { UserPlus, BookOpen, Brain, Trophy } from "lucide-react";
import { Link } from "react-router-dom";

const steps = [
  {
    icon: UserPlus,
    title: "Регистрация",
    description: "Создайте аккаунт и выберите предпочитаемый жестовый язык",
    step: "01"
  },
  {
    icon: BookOpen,
    title: "Выберите курс",
    description: "Из каталога выберите интересующий предмет и уровень сложности",
    step: "02"
  },
  {
    icon: Brain,
    title: "Обучение с AI",
    description: "Изучайте материал с помощью 3D-аватара и задавайте вопросы",
    step: "03"
  },
  {
    icon: Trophy,
    title: "Получите сертификат",
    description: "Завершите курс и получите официальный сертификат об образовании",
    step: "04"
  }
];

const HowItWorks = () => {
  return (
    <section className="py-20 bg-background overflow-x-hidden">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6">
            Как это работает?
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Простой путь к качественному образованию всего за 4 шага
          </p>
        </div>
        
        <div className="max-w-5xl mx-auto">
          {/* Steps Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
            {steps.map((step, index) => (
              <div 
                key={index}
                className="text-center group animate-fade-in-up"
                style={{animationDelay: `${index * 0.2}s`}}
              >
                {/* Step Number */}
                <div className="relative mb-6">
                  <div className="w-20 h-20 bg-gradient-primary rounded-full flex items-center justify-center mx-auto shadow-medium group-hover:shadow-glow transition-all duration-300">
                    <step.icon className="w-8 h-8 text-white" />
                  </div>
                  <div className="absolute -top-2 -right-2 w-8 h-8 bg-accent rounded-full flex items-center justify-center shadow-medium">
                    <span className="text-white font-bold text-sm">{step.step}</span>
                  </div>
                </div>
                
                <h3 className="text-xl font-bold text-foreground mb-3">
                  {step.title}
                </h3>
                
                <p className="text-muted-foreground leading-relaxed">
                  {step.description}
                </p>
                

              </div>
            ))}
          </div>
          
          {/* CTA */}
          <div className="text-center animate-fade-in">
            <div className="bg-gradient-card rounded-3xl p-8 shadow-soft">
              <h3 className="text-2xl font-bold text-foreground mb-4">
                Готовы начать свое образовательное путешествие?
              </h3>
              <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
                Присоединяйтесь к тысячам студентов, которые уже изучают новые предметы 
                с помощью нашего AI-репетитора
              </p>
              <Button asChild size="lg" className="btn-hero text-lg px-8 py-4">
                <Link to="/register">
                  Зарегистрироваться бесплатно
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;