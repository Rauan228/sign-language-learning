import { useState } from "react";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Heart, Star, Clock, Users, Play, BookOpen } from "lucide-react";
import { Link } from "react-router-dom";

interface Course {
  id: string;
  title: string;
  description: string;
  thumbnail: string;
  price: number;
  originalPrice?: number;
  rating: number;
  reviewsCount: number;
  studentsCount: number;
  duration: string;
  level: string;
  language: string;
  category: string;
  instructor: string;
  isFree: boolean;
  isNew?: boolean;
  isBestseller?: boolean;
}

interface CourseCardProps {
  course: Course;
  onToggleFavorite?: (courseId: string) => void;
  isFavorite?: boolean;
  className?: string;
}

const CourseCard = ({ course, onToggleFavorite, isFavorite = false, className = "" }: CourseCardProps) => {
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  const handleFavoriteClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onToggleFavorite?.(course.id);
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('ru-RU', {
      style: 'currency',
      currency: 'RUB',
      minimumFractionDigits: 0
    }).format(price);
  };

  const renderStars = (rating: number) => {
    return Array.from({ length: 5 }, (_, i) => (
      <Star
        key={i}
        className={`h-4 w-4 ${
          i < Math.floor(rating)
            ? 'text-yellow-400 fill-yellow-400'
            : i < rating
            ? 'text-yellow-400 fill-yellow-400/50'
            : 'text-gray-300'
        }`}
      />
    ));
  };

  return (
    <Card className={`group hover:shadow-xl transition-all duration-300 hover:-translate-y-1 border-primary/20 overflow-hidden bg-gradient-card ${className}`}>
      {/* Thumbnail */}
      <div className="relative overflow-hidden">
        <div className="aspect-video bg-gradient-to-br from-primary/20 to-secondary/20 relative">
          {!imageError ? (
            <img
              src={course.thumbnail}
              alt={course.title}
              className={`w-full h-full object-cover transition-all duration-300 group-hover:scale-105 ${
                isImageLoaded ? 'opacity-100' : 'opacity-0'
              }`}
              onLoad={() => setIsImageLoaded(true)}
              onError={() => setImageError(true)}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <BookOpen className="h-12 w-12 text-primary/40" />
            </div>
          )}
          
          {/* Overlay with play button */}
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all duration-300 flex items-center justify-center">
            <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <div className="bg-white/90 backdrop-blur rounded-full p-3 shadow-lg">
                <Play className="h-6 w-6 text-primary" />
              </div>
            </div>
          </div>
        </div>
        
        {/* Badges */}
        <div className="absolute top-3 left-3 flex flex-col gap-2">
          {course.isNew && (
            <Badge className="bg-accent text-accent-foreground font-semibold">
              Новый
            </Badge>
          )}
          {course.isBestseller && (
            <Badge className="bg-secondary text-secondary-foreground font-semibold">
              Бестселлер
            </Badge>
          )}
          {course.isFree && (
            <Badge className="bg-success text-success-foreground font-semibold">
              Бесплатно
            </Badge>
          )}
        </div>
        
        {/* Favorite button */}
        <Button
          variant="ghost"
          size="sm"
          onClick={handleFavoriteClick}
          className={`absolute top-3 right-3 h-8 w-8 p-0 rounded-full bg-white/80 backdrop-blur hover:bg-white transition-all duration-300 ${
            isFavorite ? 'text-red-500' : 'text-gray-600 hover:text-red-500'
          }`}
        >
          <Heart className={`h-4 w-4 ${isFavorite ? 'fill-current' : ''}`} />
        </Button>
      </div>
      
      <CardHeader className="pb-3">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Badge variant="outline" className="text-xs">
              {course.level}
            </Badge>
            <Badge variant="outline" className="text-xs">
              {course.language}
            </Badge>
          </div>
          
          <h3 className="font-bold text-lg leading-tight line-clamp-2 group-hover:text-primary transition-colors">
            {course.title}
          </h3>
          
          <p className="text-sm text-muted-foreground line-clamp-2">
            {course.description}
          </p>
          
          <p className="text-sm text-primary font-medium">
            {course.instructor}
          </p>
        </div>
      </CardHeader>
      
      <CardContent className="py-3">
        <div className="space-y-3">
          {/* Rating */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              {renderStars(course.rating)}
            </div>
            <span className="text-sm font-semibold">{course.rating}</span>
            <span className="text-sm text-muted-foreground">({course.reviewsCount})</span>
          </div>
          
          {/* Stats */}
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Users className="h-4 w-4" />
              <span>{course.studentsCount.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              <span>{course.duration}</span>
            </div>
          </div>
        </div>
      </CardContent>
      
      <CardFooter className="pt-3 flex items-center justify-between">
        <div className="flex flex-col">
          {course.isFree ? (
            <span className="text-xl font-bold text-success">Бесплатно</span>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold text-foreground">
                {formatPrice(course.price)}
              </span>
              {course.originalPrice && course.originalPrice > course.price && (
                <span className="text-sm text-muted-foreground line-through">
                  {formatPrice(course.originalPrice)}
                </span>
              )}
            </div>
          )}
        </div>
        
        <Button asChild className="bg-gradient-to-r from-primary to-secondary hover:from-primary-dark hover:to-secondary-light transition-all duration-300 shadow-lg hover:shadow-glow">
          <Link to={`/course/${course.id}`}>
            Подробнее
          </Link>
        </Button>
      </CardFooter>
    </Card>
  );
};

export default CourseCard;