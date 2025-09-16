<?php

namespace Database\Seeders;

use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;
use App\Models\Course;
use App\Models\Module;
use App\Models\Lesson;

class CourseSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Get the test user as instructor
        $instructor = \App\Models\User::where('email', 'test@example.com')->first();
        
        // Create courses
        $beginnerCourse = Course::create([
            'title' => 'Основы жестового языка',
            'description' => 'Изучите основы жестового языка с нуля. Этот курс познакомит вас с базовыми жестами, алфавитом и простыми фразами.',
            'price' => 2999.00,
            'difficulty_level' => 'beginner',
            'duration_hours' => 20,
            'image' => '/placeholder-course.jpg',
            'instructor_id' => $instructor->id,
            'is_published' => true,
        ]);

        $intermediateCourse = Course::create([
            'title' => 'Продвинутый жестовый язык',
            'description' => 'Углубите свои знания жестового языка. Изучите сложные конструкции, грамматику и специальную лексику.',
            'price' => 4999.00,
            'difficulty_level' => 'intermediate',
            'duration_hours' => 35,
            'image' => '/placeholder-course.jpg',
            'instructor_id' => $instructor->id,
            'is_published' => true,
        ]);

        // Create modules for beginner course
        $module1 = Module::create([
            'course_id' => $beginnerCourse->id,
            'title' => 'Алфавит и числа',
            'description' => 'Изучение дактильного алфавита и чисел в жестовом языке',
            'order_index' => 1,
        ]);

        $module2 = Module::create([
            'course_id' => $beginnerCourse->id,
            'title' => 'Базовые жесты',
            'description' => 'Основные жесты для повседневного общения',
            'order_index' => 2,
        ]);

        $module3 = Module::create([
            'course_id' => $beginnerCourse->id,
            'title' => 'Простые фразы',
            'description' => 'Составление простых предложений и фраз',
            'order_index' => 3,
        ]);

        // Create lessons for module 1
        Lesson::create([
            'module_id' => $module1->id,
            'title' => 'Буквы А-И',
            'description' => 'Изучение первых букв дактильного алфавита',
            'content' => 'В этом уроке мы изучим жесты для букв от А до И. Каждая буква имеет свой уникальный жест.',
            'video_url' => 'https://example.com/videos/alphabet-a-i.mp4',
            'order_index' => 1,
            'duration_minutes' => 15,
        ]);

        Lesson::create([
            'module_id' => $module1->id,
            'title' => 'Буквы К-С',
            'description' => 'Продолжение изучения дактильного алфавита',
            'content' => 'Изучаем жесты для букв от К до С. Обратите внимание на правильное положение пальцев.',
            'video_url' => 'https://example.com/videos/alphabet-k-s.mp4',
            'order_index' => 2,
            'duration_minutes' => 15,
        ]);

        Lesson::create([
            'module_id' => $module1->id,
            'title' => 'Буквы Т-Я',
            'description' => 'Завершение изучения дактильного алфавита',
            'content' => 'Заключительная часть алфавита - буквы от Т до Я. Практикуйте все буквы вместе.',
            'video_url' => 'https://example.com/videos/alphabet-t-ya.mp4',
            'order_index' => 3,
            'duration_minutes' => 20,
        ]);

        Lesson::create([
            'module_id' => $module1->id,
            'title' => 'Числа 1-10',
            'description' => 'Изучение жестов для чисел от 1 до 10',
            'content' => 'Числа в жестовом языке показываются особым образом. Изучите правильные жесты для чисел.',
            'video_url' => 'https://example.com/videos/numbers-1-10.mp4',
            'order_index' => 4,
            'duration_minutes' => 10,
        ]);

        // Create lessons for module 2
        Lesson::create([
            'module_id' => $module2->id,
            'title' => 'Приветствие и знакомство',
            'description' => 'Основные жесты для приветствия и знакомства',
            'content' => 'Изучите жесты: привет, до свидания, меня зовут, как дела, спасибо.',
            'video_url' => 'https://example.com/videos/greetings.mp4',
            'order_index' => 1,
            'duration_minutes' => 20,
        ]);

        Lesson::create([
            'module_id' => $module2->id,
            'title' => 'Семья и родственники',
            'description' => 'Жесты для обозначения членов семьи',
            'content' => 'Изучите жесты: мама, папа, брат, сестра, дедушка, бабушка, семья.',
            'video_url' => 'https://example.com/videos/family.mp4',
            'order_index' => 2,
            'duration_minutes' => 25,
        ]);

        // Create modules for intermediate course
        $advancedModule1 = Module::create([
            'course_id' => $intermediateCourse->id,
            'title' => 'Грамматика жестового языка',
            'description' => 'Изучение грамматических конструкций',
            'order_index' => 1,
        ]);

        Lesson::create([
            'module_id' => $advancedModule1->id,
            'title' => 'Порядок слов в предложении',
            'description' => 'Особенности построения предложений в жестовом языке',
            'content' => 'Жестовый язык имеет свою уникальную грамматику. Изучите правильный порядок жестов.',
            'video_url' => 'https://example.com/videos/grammar-word-order.mp4',
            'order_index' => 1,
            'duration_minutes' => 30,
        ]);
    }
}
