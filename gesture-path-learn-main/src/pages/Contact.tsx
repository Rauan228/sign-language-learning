import { useState } from "react";
import Header from "@/components/Header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Mail, Phone, MapPin, MessageCircle, Send } from "lucide-react";

const Contact = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: ""
  });
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Здесь будет логика отправки сообщения
    console.log("Contact form submission:", formData);
    
    // Имитация запроса
    setTimeout(() => {
      setIsLoading(false);
      alert("Сообщение отправлено! Мы свяжемся с вами в ближайшее время.");
      setFormData({ name: "", email: "", subject: "", message: "" });
    }, 1000);
  };

  const faqItems = [
    {
      question: "Как начать изучение жестового языка?",
      answer: "Зарегистрируйтесь на платформе и выберите курс для начинающих. Наш AI-ассистент поможет вам определить подходящий уровень."
    },
    {
      question: "Какие языки жестов поддерживает платформа?",
      answer: "Мы поддерживаем русский жестовый язык (РЖЯ), американский (ASL) и британский (BSL) жестовые языки."
    },
    {
      question: "Нужно ли специальное оборудование?",
      answer: "Нет, достаточно компьютера или планшета с веб-камерой для интерактивных упражнений."
    },
    {
      question: "Есть ли бесплатные курсы?",
      answer: "Да, у нас есть базовые курсы, доступные бесплатно после регистрации."
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      {/* Hero секция */}
      <section className="py-16 bg-gradient-to-r from-primary/10 to-secondary/10">
        <div className="container text-center">
          <h1 className="text-4xl font-bold mb-4">Контакты и поддержка</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Свяжитесь с нами для получения помощи или дополнительной информации
          </p>
        </div>
      </section>

      <div className="container py-16">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Форма обратной связи */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Send className="h-5 w-5" />
                  Форма обратной связи
                </CardTitle>
                <CardDescription>
                  Отправьте нам сообщение, и мы ответим в течение 24 часов
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Имя</Label>
                      <Input
                        id="name"
                        placeholder="Ваше имя"
                        value={formData.name}
                        onChange={(e) => handleInputChange("name", e.target.value)}
                        required
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email"
                        type="email"
                        placeholder="your@email.com"
                        value={formData.email}
                        onChange={(e) => handleInputChange("email", e.target.value)}
                        required
                      />
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="subject">Тема обращения</Label>
                    <Select onValueChange={(value) => handleInputChange("subject", value)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Выберите тему" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="technical">Техническая поддержка</SelectItem>
                        <SelectItem value="courses">Вопросы по курсам</SelectItem>
                        <SelectItem value="billing">Оплата и подписка</SelectItem>
                        <SelectItem value="partnership">Сотрудничество</SelectItem>
                        <SelectItem value="other">Другое</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="message">Сообщение</Label>
                    <Textarea
                      id="message"
                      placeholder="Опишите ваш вопрос или предложение..."
                      rows={5}
                      value={formData.message}
                      onChange={(e) => handleInputChange("message", e.target.value)}
                      required
                    />
                  </div>
                  
                  <Button type="submit" className="w-full" disabled={isLoading}>
                    {isLoading ? "Отправка..." : "Отправить сообщение"}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>

          {/* Контактная информация */}
          <div className="space-y-6">
            {/* Контакты */}
            <Card>
              <CardHeader>
                <CardTitle>Контактная информация</CardTitle>
                <CardDescription>
                  Свяжитесь с нами удобным для вас способом
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-3">
                  <Mail className="h-5 w-5 text-primary" />
                  <div>
                    <p className="font-medium">Email</p>
                    <p className="text-sm text-muted-foreground">support@gesturepathlearn.ru</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <Phone className="h-5 w-5 text-primary" />
                  <div>
                    <p className="font-medium">Телефон</p>
                    <p className="text-sm text-muted-foreground">+7 (495) 123-45-67</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <MapPin className="h-5 w-5 text-primary" />
                  <div>
                    <p className="font-medium">Адрес</p>
                    <p className="text-sm text-muted-foreground">
                      г. Москва, ул. Примерная, д. 123, офис 456
                    </p>
                  </div>
                </div>
                
                <Separator />
                
                <div>
                  <p className="font-medium mb-2">Время работы поддержки:</p>
                  <p className="text-sm text-muted-foreground">
                    Понедельник - Пятница: 9:00 - 18:00 (МСК)<br />
                    Суббота - Воскресенье: 10:00 - 16:00 (МСК)
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Мессенджеры */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageCircle className="h-5 w-5" />
                  Мессенджеры и соцсети
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full justify-start">
                  Telegram: @gesturepathlearn
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  WhatsApp: +7 (495) 123-45-67
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  VK: vk.com/gesturepathlearn
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* FAQ секция */}
        <section className="mt-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Часто задаваемые вопросы</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Ответы на самые популярные вопросы о нашей платформе
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {faqItems.map((item, index) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="text-lg">{item.question}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{item.answer}</p>
                </CardContent>
              </Card>
            ))}
          </div>
          
          <div className="text-center mt-8">
            <p className="text-muted-foreground mb-4">
              Не нашли ответ на свой вопрос?
            </p>
            <Button variant="outline">
              Посмотреть полный FAQ
            </Button>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Contact;