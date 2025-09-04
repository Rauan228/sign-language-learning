import { Button } from "@/components/ui/button";
import { NavigationMenu, NavigationMenuContent, NavigationMenuItem, NavigationMenuLink, NavigationMenuList, NavigationMenuTrigger } from "@/components/ui/navigation-menu";
import { Link } from "react-router-dom";

const Header = () => {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-primary/20 bg-gradient-to-r from-background/95 to-muted/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 shadow-md">
      <div className="container flex h-16 items-center justify-between">
        {/* Логотип и название */}
        <Link to="/" className="flex items-center space-x-2">
          <img 
            src="/logo.png" 
            alt="Visual Mind" 
            className="h-8 w-8"
          />
          <span className="text-xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">Visual Mind</span>
        </Link>

        {/* Навигация */}
        <NavigationMenu className="hidden md:flex">
          <NavigationMenuList>
            <NavigationMenuItem>
              <NavigationMenuTrigger>Курсы</NavigationMenuTrigger>
              <NavigationMenuContent>
                <div className="grid gap-3 p-6 w-[400px]">
                  <NavigationMenuLink asChild>
                    <Link to="/courses" className="block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                      <div className="text-sm font-medium leading-none">Каталог курсов</div>
                      <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                        Изучайте жестовый язык с помощью интерактивных курсов
                      </p>
                    </Link>
                  </NavigationMenuLink>
                  <NavigationMenuLink asChild>
                    <Link to="/my-courses" className="block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground">
                      <div className="text-sm font-medium leading-none">Мои курсы</div>
                      <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                        Продолжите изучение начатых курсов
                      </p>
                    </Link>
                  </NavigationMenuLink>
                </div>
              </NavigationMenuContent>
            </NavigationMenuItem>
            
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link to="/ai-assistant" className="group inline-flex h-10 w-max items-center justify-center rounded-md bg-background px-4 py-2 text-sm font-medium transition-all duration-300 hover:bg-gradient-to-r hover:from-primary/10 hover:to-secondary/10 hover:text-primary focus:bg-gradient-to-r focus:from-primary/10 focus:to-secondary/10 focus:text-primary focus:outline-none disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden">
                  <span className="relative z-10">AI-Ассистент</span>
                  <div className="absolute inset-0 bg-gradient-to-r from-primary to-secondary opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>
            
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link to="/about" className="group inline-flex h-10 w-max items-center justify-center rounded-md bg-background px-4 py-2 text-sm font-medium transition-all duration-300 hover:bg-gradient-to-r hover:from-primary/10 hover:to-secondary/10 hover:text-primary focus:bg-gradient-to-r focus:from-primary/10 focus:to-secondary/10 focus:text-primary focus:outline-none disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden">
                  <span className="relative z-10">О платформе</span>
                  <div className="absolute inset-0 bg-gradient-to-r from-primary to-secondary opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>
            
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link to="/contact" className="group inline-flex h-10 w-max items-center justify-center rounded-md bg-background px-4 py-2 text-sm font-medium transition-all duration-300 hover:bg-gradient-to-r hover:from-primary/10 hover:to-secondary/10 hover:text-primary focus:bg-gradient-to-r focus:from-primary/10 focus:to-secondary/10 focus:text-primary focus:outline-none disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden">
                  <span className="relative z-10">Контакты</span>
                  <div className="absolute inset-0 bg-gradient-to-r from-primary to-secondary opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>
          </NavigationMenuList>
        </NavigationMenu>

        {/* Кнопки входа и регистрации */}
        <div className="flex items-center space-x-2">
          <Button variant="ghost" asChild className="hover:bg-primary/10 hover:text-primary transition-all duration-300">
            <Link to="/login">Войти</Link>
          </Button>
          <Button asChild className="bg-gradient-to-r from-primary to-secondary hover:from-primary-dark hover:to-secondary-light transition-all duration-300 shadow-lg hover:shadow-glow">
            <Link to="/register">Регистрация</Link>
          </Button>
        </div>
      </div>
    </header>
  );
};

export default Header;