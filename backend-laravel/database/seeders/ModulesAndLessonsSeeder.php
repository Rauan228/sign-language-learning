<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use App\Models\Course;
use App\Models\Module;
use App\Models\Lesson;

class ModulesAndLessonsSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Получаем существующие курсы или создаем тестовые
        $courses = Course::all();
        
        if ($courses->isEmpty()) {
            // Создаем тестовые курсы если их нет
            $course1 = Course::create([
                'title' => 'Основы жестового языка',
                'description' => 'Изучите основы жестового языка с помощью 3D-аватара',
                'price' => 2999.00,
                'duration_hours' => 10,
                'level' => 'beginner',
                'category' => 'language',
                'teacher_name' => 'Анна Петрова',
                'is_free' => false,
                'is_published' => true,
                'thumbnail_url' => '/images/course-1.jpg',
                'rating' => 4.8,
                'students_count' => 150
            ]);
            
            $course2 = Course::create([
                'title' => 'Повседневные фразы на жестовом языке',
                'description' => 'Практические фразы для ежедневного общения',
                'price' => 1999.00,
                'duration_hours' => 6,
                'level' => 'intermediate',
                'category' => 'language',
                'teacher_name' => 'Михаил Иванов',
                'is_free' => false,
                'is_published' => true,
                'thumbnail_url' => '/images/course-2.jpg',
                'rating' => 4.6,
                'students_count' => 89
            ]);
            
            $courses = collect([$course1, $course2]);
        }
        
        foreach ($courses as $course) {
            // Создаем модули для каждого курса
            if ($course->title === 'Основы жестового языка') {
                $this->createBasicCourseModules($course);
            } elseif ($course->title === 'Повседневные фразы на жестовом языке') {
                $this->createPhraseCourseModules($course);
            }
        }
    }
    
    private function createBasicCourseModules($course)
    {
        // Модуль 1: Приветствие
        $module1 = Module::create([
            'course_id' => $course->id,
            'title' => 'Приветствие и знакомство',
            'description' => 'Основные жесты для приветствия и знакомства',
            'order_index' => 1,
            'is_published' => true
        ]);
        
        // Уроки модуля 1
        Lesson::create([
            'module_id' => $module1->id,
            'title' => 'Жест "Привет"',
            'description' => 'Изучаем базовый жест приветствия',
            'type' => 'video',
            'duration_minutes' => 5,
            'order_index' => 1,
            'is_published' => true,
            'video_url' => '/videos/lesson-hello.mp4',
            'content' => '<p>В этом уроке мы изучим основной жест приветствия "Привет". Этот жест является одним из самых важных в жестовом языке.</p>'
        ]);
        
        Lesson::create([
            'module_id' => $module1->id,
            'title' => 'Жест "Как дела?"',
            'description' => 'Учимся спрашивать о самочувствии',
            'type' => 'video',
            'duration_minutes' => 7,
            'order_index' => 2,
            'is_published' => true,
            'video_url' => '/videos/lesson-how-are-you.mp4',
            'content' => '<p>Изучаем жест "Как дела?" - важный элемент вежливого общения.</p>'
        ]);
        
        Lesson::create([
            'module_id' => $module1->id,
            'title' => 'Жест "Меня зовут..."',
            'description' => 'Представляемся на жестовом языке',
            'type' => 'video',
            'duration_minutes' => 8,
            'order_index' => 3,
            'is_published' => true,
            'video_url' => '/videos/lesson-my-name.mp4',
            'content' => '<p>Учимся представляться и называть свое имя с помощью жестов.</p>'
        ]);
        
        // Модуль 2: Простые фразы
        $module2 = Module::create([
            'course_id' => $course->id,
            'title' => 'Простые фразы',
            'description' => 'Базовые фразы для повседневного общения',
            'order_index' => 2,
            'is_published' => true
        ]);
        
        // Уроки модуля 2
        Lesson::create([
            'module_id' => $module2->id,
            'title' => 'Жесты "Да" и "Нет"',
            'description' => 'Основные жесты согласия и отрицания',
            'type' => 'video',
            'duration_minutes' => 6,
            'order_index' => 1,
            'is_published' => true,
            'video_url' => '/videos/lesson-yes-no.mp4',
            'content' => '<p>Изучаем базовые жесты "Да" и "Нет" - основа любого диалога.</p>'
        ]);
        
        Lesson::create([
            'module_id' => $module2->id,
            'title' => 'Жест "Спасибо"',
            'description' => 'Выражаем благодарность жестами',
            'type' => 'video',
            'duration_minutes' => 5,
            'order_index' => 2,
            'is_published' => true,
            'video_url' => '/videos/lesson-thank-you.mp4',
            'content' => '<p>Учимся благодарить на жестовом языке.</p>'
        ]);
        
        Lesson::create([
            'module_id' => $module2->id,
            'title' => 'Жест "Пожалуйста"',
            'description' => 'Вежливые просьбы на жестовом языке',
            'type' => 'video',
            'duration_minutes' => 6,
            'order_index' => 3,
            'is_published' => true,
            'video_url' => '/videos/lesson-please.mp4',
            'content' => '<p>Изучаем жест "Пожалуйста" для вежливых просьб.</p>'
        ]);
    }
    
    private function createPhraseCourseModules($course)
    {
        // Модуль 1: В магазине
        $module1 = Module::create([
            'course_id' => $course->id,
            'title' => 'Фразы для магазина',
            'description' => 'Полезные фразы для похода в магазин',
            'order_index' => 1,
            'is_published' => true
        ]);
        
        Lesson::create([
            'module_id' => $module1->id,
            'title' => '"Сколько это стоит?"',
            'description' => 'Спрашиваем о цене товара',
            'type' => 'video',
            'duration_minutes' => 7,
            'order_index' => 1,
            'is_published' => true,
            'video_url' => '/videos/lesson-price.mp4',
            'content' => '<p>Учимся спрашивать о цене товаров в магазине.</p>'
        ]);
        
        Lesson::create([
            'module_id' => $module1->id,
            'title' => '"Где касса?"',
            'description' => 'Ищем кассу в магазине',
            'type' => 'video',
            'duration_minutes' => 5,
            'order_index' => 2,
            'is_published' => true,
            'video_url' => '/videos/lesson-cashier.mp4',
            'content' => '<p>Изучаем, как спросить о местонахождении кассы.</p>'
        ]);
        
        // Модуль 2: В кафе
        $module2 = Module::create([
            'course_id' => $course->id,
            'title' => 'Фразы для кафе',
            'description' => 'Общение в кафе и ресторане',
            'order_index' => 2,
            'is_published' => true
        ]);
        
        Lesson::create([
            'module_id' => $module2->id,
            'title' => '"Можно меню?"',
            'description' => 'Просим меню в кафе',
            'type' => 'video',
            'duration_minutes' => 6,
            'order_index' => 1,
            'is_published' => true,
            'video_url' => '/videos/lesson-menu.mp4',
            'content' => '<p>Учимся просить меню в кафе или ресторане.</p>'
        ]);
        
        Lesson::create([
            'module_id' => $module2->id,
            'title' => '"Счет, пожалуйста"',
            'description' => 'Просим счет после еды',
            'type' => 'video',
            'duration_minutes' => 5,
            'order_index' => 2,
            'is_published' => true,
            'video_url' => '/videos/lesson-bill.mp4',
            'content' => '<p>Изучаем, как попросить счет после еды.</p>'
        ]);
    }
}