import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, X } from "lucide-react";

interface SearchBarProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  className?: string;
}

const SearchBar = ({ onSearch, placeholder = "Поиск курсов...", className = "" }: SearchBarProps) => {
  const [query, setQuery] = useState("");

  const handleSearch = () => {
    onSearch(query);
  };

  const handleClear = () => {
    setQuery("");
    onSearch("");
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className={`relative max-w-2xl mx-auto ${className}`}>
      <div className="relative">
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground h-5 w-5" />
        <Input
          type="text"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          className="pl-12 pr-20 h-14 text-lg bg-background/80 backdrop-blur border-primary/20 focus:border-primary focus:ring-2 focus:ring-primary/20 rounded-xl shadow-lg transition-all duration-300"
        />
        {query && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClear}
            className="absolute right-14 top-1/2 transform -translate-y-1/2 h-8 w-8 p-0 hover:bg-destructive/10 hover:text-destructive"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
        <Button
          onClick={handleSearch}
          className="absolute right-2 top-1/2 transform -translate-y-1/2 h-10 px-6 bg-gradient-to-r from-primary to-secondary hover:from-primary-dark hover:to-secondary-light transition-all duration-300 shadow-lg hover:shadow-glow"
        >
          Найти
        </Button>
      </div>
      
      {/* Search suggestions */}
      {query && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-background/95 backdrop-blur border border-primary/20 rounded-xl shadow-lg z-10 max-h-60 overflow-y-auto">
          <div className="p-4">
            <p className="text-sm text-muted-foreground mb-2">Популярные запросы:</p>
            <div className="flex flex-wrap gap-2">
              {["Основы жестового языка", "Русский жестовый язык", "ASL", "Дактилология", "Грамматика"].map((suggestion) => (
                <Button
                  key={suggestion}
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setQuery(suggestion);
                    onSearch(suggestion);
                  }}
                  className="text-xs hover:bg-primary/10 hover:text-primary hover:border-primary/30 transition-all duration-300"
                >
                  {suggestion}
                </Button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchBar;