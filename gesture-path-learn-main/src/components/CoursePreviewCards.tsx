import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Star, Clock, Users, ArrowRight } from "lucide-react";
import mathImage from "@/assets/course-math.jpg";
import biologyImage from "@/assets/course-biology.jpg";
import anatomyImage from "@/assets/course-anatomy.jpg";

const courses = [
  {
    id: 1,
    title: "Математика",
    description: "Алгебра и геометрия с визуальными примерами и 3D-моделями",
    image: mathImage,
    price: "Бесплатно",
    rating: 4.9,
    students: 2850,
    duration: "12 недель",
    level: "Школьная программа",
    isFree: true
  },
  {
    id: 2,
    title: "Биология",
    description: "Изучение живых организмов через интерактивные модели",
    image: biologyImage,
    price: "8,500 ₸",
    rating: 4.8,
    students: 1920,
    duration: "10 недель",
    level: "Базовый",
    isFree: false
  },
  {
    id: 3,
    title: "Анатомия человека",
    description: "3D-модели органов и систем с детальными объяснениями",
    image: anatomyImage,
    price: "12,000 ₸",
    rating: 4.9,
    students: 1456,
    duration: "16 недель",
    level: "Университетский",
    isFree: false
  }
];

const CoursePreviewCards = () => {
  return (
    <section className="py-20 bg-background">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6">
            Популярные курсы
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Выберите из более чем 50 курсов, адаптированных для визуального обучения
          </p>
        </div>
        
        <div className="grid lg:grid-cols-3 gap-8 mb-12">
          {courses.map((course, index) => (
            <div 
              key={course.id}
              className="bg-card rounded-3xl overflow-hidden shadow-soft hover:shadow-strong transition-all duration-300 hover:scale-105 animate-fade-in-up card-hover"
              style={{animationDelay: `${index * 0.2}s`}}
            >
              {/* Course Image */}
              <div className="relative h-48 overflow-hidden">
                <img 
                  src={course.image} 
                  alt={course.title}
                  className="w-full h-full object-cover transition-transform duration-300 hover:scale-110"
                />
                <div className="absolute top-4 left-4">
                  {course.isFree ? (
                    <Badge className="bg-success text-success-foreground">
                      Бесплатно
                    </Badge>
                  ) : (
                    <Badge className="bg-primary text-primary-foreground">
                      Премиум
                    </Badge>
                  )}
                </div>
                <div className="absolute top-4 right-4">
                  <Badge variant="secondary" className="bg-white/90 text-muted-dark">
                    {course.level}
                  </Badge>
                </div>
              </div>
              
              {/* Course Content */}
              <div className="p-6">
                <h3 className="text-2xl font-bold text-card-foreground mb-3">
                  {course.title}
                </h3>
                
                <p className="text-muted-foreground mb-4 line-clamp-2">
                  {course.description}
                </p>
                
                {/* Course Stats */}
                <div className="flex items-center gap-4 text-sm text-muted-foreground mb-4">
                  <div className="flex items-center gap-1">
                    <Star className="w-4 h-4 text-accent fill-current" />
                    <span className="font-semibold">{course.rating}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Users className="w-4 h-4" />
                    <span>{course.students.toLocaleString()}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    <span>{course.duration}</span>
                  </div>
                </div>
                
                {/* Price and Action */}
                <div className="flex items-center justify-between pt-4 border-t border-border">
                  <div className="text-2xl font-bold text-primary">
                    {course.price}
                  </div>
                  <Button className="btn-hero group">
                    Начать курс
                    <ArrowRight className="ml-2 w-4 h-4 transition-transform group-hover:translate-x-1" />
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        {/* View All Courses Button */}
        <div className="text-center animate-fade-in">
          <Button size="lg" variant="outline" className="text-lg px-8 py-4">
            Посмотреть все курсы
            <ArrowRight className="ml-2 w-5 h-5" />
          </Button>
        </div>
      </div>
    </section>
  );
};

export default CoursePreviewCards;