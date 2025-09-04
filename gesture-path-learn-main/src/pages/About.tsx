import Header from "@/components/Header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Heart, Users, Award, Target } from "lucide-react";
import { Link } from "react-router-dom";

const About = () => {
  const teamMembers = [
    {
      name: "Анна Петрова",
      role: "Основатель и CEO",
      description: "Эксперт в области жестового языка с 15-летним опытом преподавания",
      image: "/api/placeholder/150/150"
    },
    {
      name: "Михаил Сидоров",
      role: "Технический директор",
      description: "Разработчик AI-технологий и 3D-визуализации",
      image: "/api/placeholder/150/150"
    },
    {
      name: "Елена Козлова",
      role: "Методист",
      description: "Специалист по разработке образовательных программ",
      image: "/api/placeholder/150/150"
    }
  ];

  const partners = [
    "Российское общество глухих",
    "Московский педагогический государственный университет",
    "Центр развития жестового языка",
    "Фонд поддержки людей с нарушениями слуха"
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      {/* Hero секция */}
      <section className="py-16 bg-gradient-to-r from-primary/10 to-secondary/10">
        <div className="container text-center">
          <h1 className="text-4xl font-bold mb-4">О платформе Gesture Path Learn</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Инновационная образовательная платформа для изучения жестового языка 
            с использованием 3D-аватаров и искусственного интеллекта
          </p>
        </div>
      </section>

      <div className="container py-16 space-y-16">
        {/* Описание проекта */}
        <section>
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl flex items-center gap-2">
                <Target className="h-6 w-6 text-primary" />
                О проекте
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-lg">
                Gesture Path Learn — это современная образовательная платформа, созданная для 
                эффективного изучения жестового языка. Мы используем передовые технологии 
                3D-визуализации и искусственного интеллекта, чтобы сделать процесс обучения 
                максимально интерактивным и доступным.
              </p>
              <p>
                Наша платформа предоставляет уникальную возможность изучать жестовый язык 
                через взаимодействие с реалистичными 3D-аватарами, которые демонстрируют 
                правильное выполнение жестов и помогают развивать навыки коммуникации.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* Миссия и ценности */}
        <section>
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Наша миссия и ценности</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Мы стремимся сделать жестовый язык доступным для всех и разрушить барьеры в общении
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="text-center">
              <CardHeader>
                <Heart className="h-12 w-12 text-primary mx-auto mb-4" />
                <CardTitle>Инклюзивность</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Создаем равные возможности для всех людей, независимо от их слуховых способностей
                </p>
              </CardContent>
            </Card>
            
            <Card className="text-center">
              <CardHeader>
                <Users className="h-12 w-12 text-primary mx-auto mb-4" />
                <CardTitle>Сообщество</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Объединяем людей через изучение жестового языка и создаем поддерживающее сообщество
                </p>
              </CardContent>
            </Card>
            
            <Card className="text-center">
              <CardHeader>
                <Award className="h-12 w-12 text-primary mx-auto mb-4" />
                <CardTitle>Качество</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Предоставляем высококачественное образование с использованием современных технологий
                </p>
              </CardContent>
            </Card>
            
            <Card className="text-center">
              <CardHeader>
                <Target className="h-12 w-12 text-primary mx-auto mb-4" />
                <CardTitle>Инновации</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Постоянно развиваем технологии для улучшения процесса изучения жестового языка
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Команда */}
        <section>
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Наша команда</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Профессионалы, объединенные общей целью — сделать жестовый язык доступным для всех
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <Card key={index} className="text-center">
                <CardHeader>
                  <img 
                    src={member.image} 
                    alt={member.name}
                    className="w-24 h-24 rounded-full mx-auto mb-4 object-cover"
                  />
                  <CardTitle>{member.name}</CardTitle>
                  <CardDescription>
                    <Badge variant="secondary">{member.role}</Badge>
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">{member.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Партнеры */}
        <section>
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Наши партнеры</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Мы сотрудничаем с ведущими организациями в области образования и поддержки людей с нарушениями слуха
            </p>
          </div>
          
          <Card>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {partners.map((partner, index) => (
                  <div key={index} className="flex items-center p-4 border rounded-lg">
                    <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mr-4">
                      <Users className="h-6 w-6 text-primary" />
                    </div>
                    <span className="font-medium">{partner}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </section>

        {/* CTA секция */}
        <section className="text-center">
          <Card className="bg-gradient-to-r from-primary/10 to-secondary/10">
            <CardContent className="pt-8 pb-8">
              <h2 className="text-2xl font-bold mb-4">Присоединяйтесь к нам</h2>
              <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
                Начните изучение жестового языка уже сегодня и станьте частью нашего сообщества
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button asChild size="lg">
                  <Link to="/register">Зарегистрироваться</Link>
                </Button>
                <Button variant="outline" size="lg" asChild>
                  <Link to="/courses">Посмотреть курсы</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
};

export default About;