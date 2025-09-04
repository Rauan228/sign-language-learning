import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Search, Star, Heart, Clock, Users } from "lucide-react";
import Header from "@/components/Header";

// Моковые данные курсов
const mockCourses = [
  {
    id: 1,
    title: "Основы русского жестового языка",
    description: "Изучите базовые жесты и грамматику РЖЯ. Курс для начинающих с интерактивными уроками.",
    image: "/api/placeholder/300/200",
    rating: 4.8,
    students: 1250,
    duration: "8 недель",
    level: "Начинающий",
    language: "РЖЯ",
    category: "Основы",
    price: "Бесплатно",
    isFavorite: false
  },
  {
    id: 2,
    title: "Математика на жестовом языке",
    description: "Изучайте математические концепции через жестовый язык. Подходит для школьников и студентов.",
    image: "/api/placeholder/300/200",
    rating: 4.6,
    students: 890,
    duration: "12 недель",
    level: "Средний",
    language: "РЖЯ",
    category: "Предметы",
    price: "1990 ₽",
    isFavorite: true
  },
  {
    id: 3,
    title: "Деловое общение на ASL",
    description: "Профессиональное общение на американском жестовом языке для бизнес-среды.",
    image: "/api/placeholder/300/200",
    rating: 4.9,
    students: 567,
    duration: "6 недель",
    level: "Продвинутый",
    language: "ASL",
    category: "Бизнес",
    price: "2990 ₽",
    isFavorite: false
  },
  {
    id: 4,
    title: "История на жестовом языке",
    description: "Изучение исторических событий и понятий через жестовый язык.",
    image: "/api/placeholder/300/200",
    rating: 4.7,
    students: 432,
    duration: "10 недель",
    level: "Средний",
    language: "РЖЯ",
    category: "Предметы",
    price: "1590 ₽",
    isFavorite: false
  }
];

const Courses = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedLevel, setSelectedLevel] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const [courses, setCourses] = useState(mockCourses);

  const toggleFavorite = (courseId: number) => {
    setCourses(prev => 
      prev.map(course => 
        course.id === courseId 
          ? { ...course, isFavorite: !course.isFavorite }
          : course
      )
    );
  };

  const filteredCourses = courses.filter(course => {
    const matchesSearch = course.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         course.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesLevel = !selectedLevel || course.level === selectedLevel;
    const matchesLanguage = !selectedLanguage || course.language === selectedLanguage;
    const matchesCategory = !selectedCategory || course.category === selectedCategory;
    const matchesFavorites = !showFavoritesOnly || course.isFavorite;
    
    return matchesSearch && matchesLevel && matchesLanguage && matchesCategory && matchesFavorites;
  });

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Каталог курсов</h1>
          <p className="text-muted-foreground">
            Изучайте жестовый язык с помощью интерактивных курсов и 3D-аватаров
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Фильтры */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Фильтры</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Поиск */}
                <div className="space-y-2">
                  <Label>Поиск курсов</Label>
                  <div className="relative">
                    <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Название или описание..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>

                <Separator />

                {/* Уровень */}
                <div className="space-y-2">
                  <Label>Уровень сложности</Label>
                  <Select value={selectedLevel} onValueChange={setSelectedLevel}>
                    <SelectTrigger>
                      <SelectValue placeholder="Все уровни" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">Все уровни</SelectItem>
                      <SelectItem value="Начинающий">Начинающий</SelectItem>
                      <SelectItem value="Средний">Средний</SelectItem>
                      <SelectItem value="Продвинутый">Продвинутый</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Язык жеста */}
                <div className="space-y-2">
                  <Label>Язык жеста</Label>
                  <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                    <SelectTrigger>
                      <SelectValue placeholder="Все языки" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">Все языки</SelectItem>
                      <SelectItem value="РЖЯ">Русский жестовый язык</SelectItem>
                      <SelectItem value="ASL">Американский жестовый язык</SelectItem>
                      <SelectItem value="BSL">Британский жестовый язык</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Категория */}
                <div className="space-y-2">
                  <Label>Категория</Label>
                  <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                    <SelectTrigger>
                      <SelectValue placeholder="Все категории" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">Все категории</SelectItem>
                      <SelectItem value="Основы">Основы</SelectItem>
                      <SelectItem value="Предметы">Предметы</SelectItem>
                      <SelectItem value="Бизнес">Бизнес</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Separator />

                {/* Избранное */}
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="favorites" 
                    checked={showFavoritesOnly}
                    onCheckedChange={(checked) => setShowFavoritesOnly(checked as boolean)}
                  />
                  <Label htmlFor="favorites">Только избранные</Label>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Список курсов */}
          <div className="lg:col-span-3">
            <div className="mb-4 flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                Найдено курсов: {filteredCourses.length}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {filteredCourses.map((course) => (
                <Card key={course.id} className="overflow-hidden hover:shadow-lg transition-shadow">
                  <div className="relative">
                    <img 
                      src={course.image} 
                      alt={course.title}
                      className="w-full h-48 object-cover"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute top-2 right-2 bg-white/80 hover:bg-white"
                      onClick={() => toggleFavorite(course.id)}
                    >
                      <Heart 
                        className={`h-4 w-4 ${
                          course.isFavorite 
                            ? 'fill-red-500 text-red-500' 
                            : 'text-gray-600'
                        }`} 
                      />
                    </Button>
                  </div>
                  
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <CardTitle className="text-lg line-clamp-2">{course.title}</CardTitle>
                    </div>
                    <CardDescription className="line-clamp-3">
                      {course.description}
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="pb-3">
                    <div className="flex flex-wrap gap-2 mb-3">
                      <Badge variant="secondary">{course.level}</Badge>
                      <Badge variant="outline">{course.language}</Badge>
                      <Badge variant="outline">{course.category}</Badge>
                    </div>
                    
                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                        <span>{course.rating}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Users className="h-4 w-4" />
                        <span>{course.students}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Clock className="h-4 w-4" />
                        <span>{course.duration}</span>
                      </div>
                    </div>
                  </CardContent>
                  
                  <CardFooter className="pt-0">
                    <div className="flex items-center justify-between w-full">
                      <span className="text-lg font-semibold">{course.price}</span>
                      <Button>Начать изучение</Button>
                    </div>
                  </CardFooter>
                </Card>
              ))}
            </div>

            {filteredCourses.length === 0 && (
              <div className="text-center py-12">
                <p className="text-muted-foreground mb-4">
                  По вашему запросу курсы не найдены
                </p>
                <Button 
                  variant="outline" 
                  onClick={() => {
                    setSearchQuery("");
                    setSelectedLevel("");
                    setSelectedLanguage("");
                    setSelectedCategory("");
                    setShowFavoritesOnly(false);
                  }}
                >
                  Сбросить фильтры
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Courses;