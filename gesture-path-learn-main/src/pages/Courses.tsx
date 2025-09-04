import { useState, useEffect } from "react";
import Header from "@/components/Header";
import SearchBar from "@/components/SearchBar";
import FiltersPanel from "@/components/FiltersPanel";
import CourseCard from "@/components/CourseCard";
import Pagination from "@/components/Pagination";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Filter, SlidersHorizontal, Grid3X3, List } from "lucide-react";

// Моковые данные курсов
  const mockCourses = [
    {
      id: "1",
      title: "Основы русского жестового языка",
      description: "Изучите базовые жесты и грамматику РЖЯ с нуля. Курс включает интерактивные упражнения с 3D-аватаром.",
      thumbnail: "/api/placeholder/400/225",
      price: 2500,
      originalPrice: 3500,
      rating: 4.8,
      reviewsCount: 156,
      studentsCount: 1250,
      duration: "8 недель",
      level: "Начинающий",
      language: "Русский жестовый язык (РЖЯ)",
      category: "Основы жестового языка",
      instructor: "Анна Петрова",
      isFree: false,
      isNew: true,
      isBestseller: true
    },
    {
      id: "2",
      title: "Дактилология для начинающих",
      description: "Освойте пальцевую азбуку и научитесь быстро дактилировать. Практические упражнения и тесты.",
      thumbnail: "/api/placeholder/400/225",
      price: 1500,
      rating: 4.6,
      reviewsCount: 89,
      studentsCount: 890,
      duration: "4 недели",
      level: "Начинающий",
      language: "Русский жестовый язык (РЖЯ)",
      category: "Дактилология",
      instructor: "Михаил Сидоров",
      isFree: false
    },
    {
      id: "3",
      title: "American Sign Language (ASL) - Intermediate",
      description: "Продвинутый курс американского жестового языка с изучением сложной грамматики.",
      thumbnail: "/api/placeholder/400/225",
      price: 3500,
      rating: 4.9,
      reviewsCount: 234,
      studentsCount: 650,
      duration: "12 недель",
      level: "Средний",
      language: "American Sign Language (ASL)",
      category: "Грамматика",
      instructor: "Sarah Johnson",
      isFree: false,
      isBestseller: true
    },
    {
      id: "4",
      title: "Культура глухих и слабослышащих",
      description: "Изучите историю и культуру сообщества глухих. Бесплатный вводный курс.",
      thumbnail: "/api/placeholder/400/225",
      price: 0,
      rating: 4.7,
      reviewsCount: 312,
      studentsCount: 2100,
      duration: "6 недель",
      level: "Начинающий",
      language: "Русский жестовый язык (РЖЯ)",
      category: "Культура глухих",
      instructor: "Елена Козлова",
      isFree: true,
      isNew: true
    },
    {
      id: "5",
      title: "Профессиональная лексика в медицине",
      description: "Специализированные жесты для медицинских работников и студентов медвузов.",
      thumbnail: "/api/placeholder/400/225",
      price: 4000,
      rating: 4.5,
      reviewsCount: 67,
      studentsCount: 320,
      duration: "10 недель",
      level: "Продвинутый",
      language: "Русский жестовый язык (РЖЯ)",
      category: "Профессиональная лексика",
      instructor: "Дмитрий Волков",
      isFree: false
    },
    {
      id: "6",
      title: "British Sign Language (BSL) Basics",
      description: "Основы британского жестового языка для начинающих с интерактивными упражнениями.",
      thumbnail: "/api/placeholder/400/225",
      price: 2800,
      rating: 4.4,
      reviewsCount: 123,
      studentsCount: 450,
      duration: "8 недель",
      level: "Начинающий",
      language: "British Sign Language (BSL)",
      category: "Основы жестового языка",
      instructor: "James Wilson",
      isFree: false
    },
    {
      id: "7",
      title: "Детский жестовый язык",
      description: "Специальный курс для обучения детей жестовому языку в игровой форме.",
      thumbnail: "/api/placeholder/400/225",
      price: 2200,
      rating: 4.8,
      reviewsCount: 98,
      studentsCount: 567,
      duration: "6 недель",
      level: "Начинающий",
      language: "Русский жестовый язык (РЖЯ)",
      category: "Детский жестовый язык",
      instructor: "Ольга Иванова",
      isFree: false,
      isNew: true
    },
    {
      id: "8",
      title: "Лексика повседневного общения",
      description: "Изучите жесты для повседневного общения: семья, работа, покупки, транспорт.",
      thumbnail: "/api/placeholder/400/225",
      price: 1800,
      rating: 4.6,
      reviewsCount: 145,
      studentsCount: 890,
      duration: "5 недель",
      level: "Базовый",
      language: "Русский жестовый язык (РЖЯ)",
      category: "Лексика",
      instructor: "Татьяна Смирнова",
      isFree: false
    }
  ];

interface FilterState {
  subjects: string[];
  levels: string[];
  languages: string[];
  priceRange: [number, number];
  rating: number;
  duration: string;
  isFree: boolean;
}

const Courses = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [filters, setFilters] = useState<FilterState>({
    subjects: [],
    levels: [],
    languages: [],
    priceRange: [0, 10000],
    rating: 0,
    duration: "",
    isFree: false
  });
  const [sortBy, setSortBy] = useState("relevance");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [showFilters, setShowFilters] = useState(false);
  const [favorites, setFavorites] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(12);
  const [courses, setCourses] = useState(mockCourses);

  // Логика фильтрации и поиска
  const filteredCourses = mockCourses.filter(course => {
    // Поиск по названию и описанию
    const matchesSearch = !searchQuery || 
      course.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      course.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      course.instructor.toLowerCase().includes(searchQuery.toLowerCase());
    
    // Фильтр по предметам
    const matchesSubjects = filters.subjects.length === 0 || 
      filters.subjects.includes(course.category);
    
    // Фильтр по уровням
    const matchesLevels = filters.levels.length === 0 || 
      filters.levels.includes(course.level);
    
    // Фильтр по языкам
    const matchesLanguages = filters.languages.length === 0 || 
      filters.languages.includes(course.language);
    
    // Фильтр по цене
    const matchesPrice = (!filters.isFree || course.isFree) &&
      course.price >= filters.priceRange[0] && course.price <= filters.priceRange[1];
    
    // Фильтр по рейтингу
    const matchesRating = filters.rating === 0 || course.rating >= filters.rating;
    
    return matchesSearch && matchesSubjects && matchesLevels && 
           matchesLanguages && matchesPrice && matchesRating;
  });

  // Сортировка
  const sortedCourses = [...filteredCourses].sort((a, b) => {
    switch (sortBy) {
      case "price-low":
        return a.price - b.price;
      case "price-high":
        return b.price - a.price;
      case "rating":
        return b.rating - a.rating;
      case "students":
        return b.studentsCount - a.studentsCount;
      case "newest":
        return a.isNew ? -1 : b.isNew ? 1 : 0;
      default:
        return 0;
    }
  });

  // Пагинация
  const totalPages = Math.ceil(sortedCourses.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedCourses = sortedCourses.slice(startIndex, startIndex + itemsPerPage);

  // Управление избранным
  const toggleFavorite = (courseId: string) => {
    setFavorites(prev => 
      prev.includes(courseId)
        ? prev.filter(id => id !== courseId)
        : [...prev, courseId]
    );
  };

  // Сброс пагинации при изменении фильтров
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, filters, sortBy]);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      {/* Hero секция */}
      <section className="py-16 bg-gradient-to-r from-primary/10 to-secondary/10">
        <div className="container text-center">
          <h1 className="text-4xl font-bold mb-4">Каталог курсов Visual Mind</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Выберите курс жестового языка, который подходит именно вам. Интерактивное обучение с AI-ассистентом.
          </p>
        </div>
      </section>

      <div className="container py-8">
        {/* Поиск */}
        <div className="mb-8">
          <SearchBar
            onSearch={setSearchQuery}
            placeholder="Поиск курсов по названию, описанию или преподавателю..."
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Фильтры */}
          <div className="lg:col-span-1">
            <div className={`lg:block ${showFilters ? 'block' : 'hidden'}`}>
              <FiltersPanel
                onFiltersChange={setFilters}
                className="sticky top-4"
              />
            </div>
          </div>

          {/* Основной контент */}
          <div className="lg:col-span-3">
            {/* Панель управления */}
            <div className="mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <Button
                  variant="outline"
                  onClick={() => setShowFilters(!showFilters)}
                  className="lg:hidden"
                >
                  <SlidersHorizontal className="h-4 w-4 mr-2" />
                  Фильтры
                </Button>
                
                <p className="text-muted-foreground">
                  Найдено <span className="font-semibold text-foreground">{sortedCourses.length}</span> курсов
                </p>
              </div>
              
              <div className="flex items-center gap-4">
                {/* Вид отображения */}
                <div className="flex items-center border border-primary/20 rounded-lg p-1">
                  <Button
                    variant={viewMode === 'grid' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setViewMode('grid')}
                    className="h-8 w-8 p-0"
                  >
                    <Grid3X3 className="h-4 w-4" />
                  </Button>
                  <Button
                    variant={viewMode === 'list' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setViewMode('list')}
                    className="h-8 w-8 p-0"
                  >
                    <List className="h-4 w-4" />
                  </Button>
                </div>
                
                {/* Сортировка */}
                <Select value={sortBy} onValueChange={setSortBy}>
                  <SelectTrigger className="w-48 border-primary/20 focus:border-primary focus:ring-primary/20">
                    <SelectValue placeholder="Сортировать по" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="relevance">По релевантности</SelectItem>
                    <SelectItem value="rating">По рейтингу</SelectItem>
                    <SelectItem value="students">По популярности</SelectItem>
                    <SelectItem value="price-low">Сначала дешевые</SelectItem>
                    <SelectItem value="price-high">Сначала дорогие</SelectItem>
                    <SelectItem value="newest">Сначала новые</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Активные фильтры */}
            {(filters.subjects.length > 0 || filters.levels.length > 0 || filters.languages.length > 0 || filters.isFree) && (
              <div className="mb-6 flex flex-wrap items-center gap-2">
                <span className="text-sm text-muted-foreground">Активные фильтры:</span>
                {filters.subjects.map(subject => (
                  <Badge key={subject} variant="secondary" className="bg-primary/10 text-primary">
                    {subject}
                  </Badge>
                ))}
                {filters.levels.map(level => (
                  <Badge key={level} variant="secondary" className="bg-secondary/10 text-secondary">
                    {level}
                  </Badge>
                ))}
                {filters.languages.map(language => (
                  <Badge key={language} variant="secondary" className="bg-accent/10 text-accent">
                    {language}
                  </Badge>
                ))}
                {filters.isFree && (
                  <Badge variant="secondary" className="bg-success/10 text-success">
                    Бесплатные
                  </Badge>
                )}
              </div>
            )}

            {/* Список курсов */}
            {paginatedCourses.length > 0 ? (
              <>
                <div className={`grid gap-6 ${
                  viewMode === 'grid' 
                    ? 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3' 
                    : 'grid-cols-1'
                }`}>
                  {paginatedCourses.map((course) => (
                    <CourseCard
                      key={course.id}
                      course={course}
                      onToggleFavorite={toggleFavorite}
                      isFavorite={favorites.includes(course.id)}
                      className={viewMode === 'list' ? 'flex-row' : ''}
                    />
                  ))}
                </div>
                
                {/* Пагинация */}
                {totalPages > 1 && (
                  <div className="mt-12">
                    <Pagination
                      currentPage={currentPage}
                      totalPages={totalPages}
                      totalItems={sortedCourses.length}
                      itemsPerPage={itemsPerPage}
                      onPageChange={setCurrentPage}
                      onItemsPerPageChange={setItemsPerPage}
                    />
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-12">
                <div className="max-w-md mx-auto">
                  <div className="mb-4">
                    <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                      <Filter className="h-8 w-8 text-muted-foreground" />
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold mb-2">Курсы не найдены</h3>
                  <p className="text-muted-foreground mb-4">
                    Попробуйте изменить параметры поиска или фильтры
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setSearchQuery("");
                      setFilters({
                        subjects: [],
                        levels: [],
                        languages: [],
                        priceRange: [0, 10000],
                        rating: 0,
                        duration: "",
                        isFree: false
                      });
                    }}
                  >
                    Сбросить фильтры
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Courses;