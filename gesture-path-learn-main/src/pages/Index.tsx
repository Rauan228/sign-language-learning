import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import FeaturesList from "@/components/FeaturesList";
import CoursePreviewCards from "@/components/CoursePreviewCards";
import DemoVideo from "@/components/DemoVideo";
import HowItWorks from "@/components/HowItWorks";
import Testimonials from "@/components/Testimonials";
import CTASection from "@/components/CTASection";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <HeroSection />
      <FeaturesList />
      <CoursePreviewCards />
      <DemoVideo />
      <HowItWorks />
      <Testimonials />
      <CTASection />
    </div>
  );
};

export default Index;