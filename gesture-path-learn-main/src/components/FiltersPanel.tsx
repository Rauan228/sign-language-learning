import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Filter, X, RotateCcw } from "lucide-react";

interface FilterState {
  subjects: string[];
  levels: string[];
  languages: string[];
  priceRange: [number, number];
  rating: number;
  duration: string;
  isFree: boolean;
}

interface FiltersPanelProps {
  onFiltersChange: (filters: FilterState) => void;
  className?: string;
}

const FiltersPanel = ({ onFiltersChange, className = "" }: FiltersPanelProps) => {
  const [filters, setFilters] = useState<FilterState>({
    subjects: [],
    levels: [],
    languages: [],
    priceRange: [0, 10000],
    rating: 0,
    duration: "",
    isFree: false
  });

  const [isExpanded, setIsExpanded] = useState(false);

  const subjects = [
    "Основы жестового языка",
    "Дактилология",
    "Грамматика",
    "Лексика",
    "Культура глухих",
    "Профессиональная лексика",
    "Детский жестовый язык"
  ];

  const levels = ["Начинающий", "Базовый", "Средний", "Продвинутый", "Эксперт"];
  const languages = ["Русский жестовый язык (РЖЯ)", "American Sign Language (ASL)", "British Sign Language (BSL)"];
  const durations = ["До 1 часа", "1-3 часа", "3-10 часов", "10+ часов"];

  const updateFilters = (newFilters: Partial<FilterState>) => {
    const updatedFilters = { ...filters, ...newFilters };
    setFilters(updatedFilters);
    onFiltersChange(updatedFilters);
  };

  const toggleArrayFilter = (category: keyof Pick<FilterState, 'subjects' | 'levels' | 'languages'>, value: string) => {
    const currentArray = filters[category];
    const newArray = currentArray.includes(value)
      ? currentArray.filter(item => item !== value)
      : [...currentArray, value];
    updateFilters({ [category]: newArray });
  };

  const clearAllFilters = () => {
    const clearedFilters: FilterState = {
      subjects: [],
      levels: [],
      languages: [],
      priceRange: [0, 10000],
      rating: 0,
      duration: "",
      isFree: false
    };
    setFilters(clearedFilters);
    onFiltersChange(clearedFilters);
  };

  const getActiveFiltersCount = () => {
    return filters.subjects.length + filters.levels.length + filters.languages.length + 
           (filters.rating > 0 ? 1 : 0) + (filters.duration ? 1 : 0) + (filters.isFree ? 1 : 0);
  };

  return (
    <Card className={`${className} border-primary/20 shadow-lg`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Filter className="h-5 w-5 text-primary" />
            Фильтры
            {getActiveFiltersCount() > 0 && (
              <Badge variant="secondary" className="bg-primary/10 text-primary">
                {getActiveFiltersCount()}
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={clearAllFilters}
              className="text-muted-foreground hover:text-destructive"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="md:hidden"
            >
              {isExpanded ? <X className="h-4 w-4" /> : <Filter className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className={`space-y-6 ${isExpanded ? 'block' : 'hidden md:block'}`}>
        {/* Предметы */}
        <div>
          <h4 className="font-semibold mb-3 text-foreground">Предмет</h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {subjects.map((subject) => (
              <div key={subject} className="flex items-center space-x-2">
                <Checkbox
                  id={subject}
                  checked={filters.subjects.includes(subject)}
                  onCheckedChange={() => toggleArrayFilter('subjects', subject)}
                  className="border-primary/30 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                />
                <label htmlFor={subject} className="text-sm cursor-pointer hover:text-primary transition-colors">
                  {subject}
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* Уровень */}
        <div>
          <h4 className="font-semibold mb-3 text-foreground">Уровень сложности</h4>
          <div className="space-y-2">
            {levels.map((level) => (
              <div key={level} className="flex items-center space-x-2">
                <Checkbox
                  id={level}
                  checked={filters.levels.includes(level)}
                  onCheckedChange={() => toggleArrayFilter('levels', level)}
                  className="border-primary/30 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                />
                <label htmlFor={level} className="text-sm cursor-pointer hover:text-primary transition-colors">
                  {level}
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* Язык жестов */}
        <div>
          <h4 className="font-semibold mb-3 text-foreground">Язык жестов</h4>
          <div className="space-y-2">
            {languages.map((language) => (
              <div key={language} className="flex items-center space-x-2">
                <Checkbox
                  id={language}
                  checked={filters.languages.includes(language)}
                  onCheckedChange={() => toggleArrayFilter('languages', language)}
                  className="border-primary/30 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                />
                <label htmlFor={language} className="text-sm cursor-pointer hover:text-primary transition-colors">
                  {language}
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* Цена */}
        <div>
          <h4 className="font-semibold mb-3 text-foreground">Цена (₽)</h4>
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="free"
                checked={filters.isFree}
                onCheckedChange={(checked) => updateFilters({ isFree: !!checked })}
                className="border-primary/30 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
              />
              <label htmlFor="free" className="text-sm cursor-pointer hover:text-primary transition-colors">
                Только бесплатные
              </label>
            </div>
            <div className="px-2">
              <Slider
                value={filters.priceRange}
                onValueChange={(value) => updateFilters({ priceRange: value as [number, number] })}
                max={10000}
                min={0}
                step={500}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>{filters.priceRange[0]} ₽</span>
                <span>{filters.priceRange[1]} ₽</span>
              </div>
            </div>
          </div>
        </div>

        {/* Продолжительность */}
        <div>
          <h4 className="font-semibold mb-3 text-foreground">Продолжительность</h4>
          <Select value={filters.duration} onValueChange={(value) => updateFilters({ duration: value })}>
            <SelectTrigger className="border-primary/20 focus:border-primary focus:ring-primary/20">
              <SelectValue placeholder="Выберите продолжительность" />
            </SelectTrigger>
            <SelectContent>
              {durations.map((duration) => (
                <SelectItem key={duration} value={duration}>
                  {duration}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Рейтинг */}
        <div>
          <h4 className="font-semibold mb-3 text-foreground">Минимальный рейтинг</h4>
          <div className="space-y-2">
            {[4, 3, 2, 1].map((rating) => (
              <div key={rating} className="flex items-center space-x-2">
                <Checkbox
                  id={`rating-${rating}`}
                  checked={filters.rating === rating}
                  onCheckedChange={() => updateFilters({ rating: filters.rating === rating ? 0 : rating })}
                  className="border-primary/30 data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                />
                <label htmlFor={`rating-${rating}`} className="text-sm cursor-pointer hover:text-primary transition-colors flex items-center">
                  {rating}+ ⭐
                </label>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default FiltersPanel;