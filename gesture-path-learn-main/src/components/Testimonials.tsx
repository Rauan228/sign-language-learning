import { Star, Quote } from "lucide-react";

const testimonials = [
  {
    name: "Айнур Касымова",
    role: "Студентка, 2 курс",
    content: "Благодаря AI-репетитору я впервые поняла алгебру! Жестовые объяснения намного понятнее обычных лекций.",
    rating: 5,
    avatar: "👩‍🎓"
  },
  {
    name: "Дмитрий Петров",
    role: "Учитель биологии",
    content: "Использую платформу для подготовки уроков. Визуальные материалы помогают объяснять сложные темы всем студентам.",
    rating: 5,
    avatar: "👨‍🏫"
  },
  {
    name: "Мария Сидорова",
    role: "Студентка медицинского",
    content: "3D-модели по анатомии просто невероятные! Теперь я могу изучать строение органов в интерактивном режиме.",
    rating: 5,
    avatar: "👩‍⚕️"
  }
];

const Testimonials = () => {
  return (
    <section className="py-20 bg-muted/30">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6">
            Отзывы студентов и преподавателей
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Узнайте, что говорят те, кто уже использует нашу платформу для обучения
          </p>
        </div>
        
        <div className="grid lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {testimonials.map((testimonial, index) => (
            <div 
              key={index}
              className="bg-card rounded-3xl p-8 shadow-soft hover:shadow-strong transition-all duration-300 hover:scale-105 animate-fade-in-up card-hover"
              style={{animationDelay: `${index * 0.2}s`}}
            >
              {/* Quote Icon */}
              <div className="mb-6">
                <Quote className="w-8 h-8 text-primary opacity-50" />
              </div>
              
              {/* Rating */}
              <div className="flex gap-1 mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <Star key={i} className="w-5 h-5 text-accent fill-current" />
                ))}
              </div>
              
              {/* Content */}
              <p className="text-card-foreground text-lg leading-relaxed mb-6 italic">
                "{testimonial.content}"
              </p>
              
              {/* Author */}
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-gradient-primary rounded-full flex items-center justify-center text-2xl">
                  {testimonial.avatar}
                </div>
                <div>
                  <h4 className="font-bold text-card-foreground">
                    {testimonial.name}
                  </h4>
                  <p className="text-muted-foreground text-sm">
                    {testimonial.role}
                  </p>
                </div>
              </div>
              
              {/* Sign Language Badge */}
              <div className="mt-4 pt-4 border-t border-border">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span className="text-lg">🤟</span>
                  <span>Отзыв доступен на жестовом языке</span>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        {/* Statistics */}
        <div className="grid md:grid-cols-3 gap-8 mt-16 max-w-4xl mx-auto">
          <div className="text-center animate-fade-in-up" style={{animationDelay: '0.6s'}}>
            <div className="text-4xl font-bold text-primary mb-2">4.9</div>
            <div className="text-muted-foreground">Средняя оценка</div>
          </div>
          <div className="text-center animate-fade-in-up" style={{animationDelay: '0.8s'}}>
            <div className="text-4xl font-bold text-secondary mb-2">10,000+</div>
            <div className="text-muted-foreground">Довольных студентов</div>
          </div>
          <div className="text-center animate-fade-in-up" style={{animationDelay: '1s'}}>
            <div className="text-4xl font-bold text-accent mb-2">95%</div>
            <div className="text-muted-foreground">Завершают курсы</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Testimonials;