import { Eye, Users, Zap, Heart, Globe, Accessibility } from "lucide-react";

const features = [
  {
    icon: Accessibility,
    title: "Жестовый язык",
    description: "Поддержка казахского и русского жестового языка с 3D-аватарами"
  },
  {
    icon: Eye,
    title: "Визуальное обучение",
    description: "Инфографика, анимации и минимум текста для лучшего восприятия"
  },
  {
    icon: Zap,
    title: "AI-репетитор",
    description: "Интеллектуальный помощник отвечает на вопросы через жесты"
  },
  {
    icon: Users,
    title: "Адаптивность",
    description: "Персонализированный подход к каждому студенту"
  },
  {
    icon: Globe,
    title: "Доступность",
    description: "Работает на всех устройствах - вебе, мобильных и в центрах"
  },
  {
    icon: Heart,
    title: "Инклюзивность",
    description: "Создано специально для людей с ограниченным слухом"
  }
];

const FeaturesList = () => {
  return (
    <section className="py-20 bg-muted/30">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6">
            Почему выбирают нашу платформу?
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Мы создали уникальную образовательную среду, специально адаптированную 
            для людей с нарушениями слуха
          </p>
        </div>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="bg-card rounded-2xl p-8 shadow-soft hover:shadow-strong transition-all duration-300 hover:scale-105 animate-fade-in-up card-hover"
              style={{animationDelay: `${index * 0.1}s`}}
            >
              <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mb-6 shadow-medium">
                <feature.icon className="w-8 h-8 text-white" />
              </div>
              
              <h3 className="text-2xl font-bold text-card-foreground mb-4">
                {feature.title}
              </h3>
              
              <p className="text-muted-foreground text-lg leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesList;