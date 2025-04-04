import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";
import { FileVideo, BarChart2, Users, ArrowRight, Database, Github } from "lucide-react";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="bg-background border-b">
        <div className="container mx-auto py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold">Thai Boxing Vision</h1>
          <nav className="flex gap-4">
            <Link href="/dashboard">
              <Button variant="ghost">Dashboard</Button>
            </Link>
            <Link href="/compare">
              <Button variant="ghost">Compare</Button>
            </Link>
            <Link href="https://github.com/your-username/thai-boxing-vision-app" target="_blank">
              <Button variant="ghost">
                <Github className="h-4 w-4 mr-2" />
                GitHub
              </Button>
            </Link>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <section className="py-20 bg-gradient-to-b from-background to-muted">
          <div className="container mx-auto text-center">
            <h1 className="text-5xl font-bold mb-6">AI-Powered Thai Boxing Analysis</h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-10">
              Track techniques, measure speed, analyze damage, and visualize match statistics with advanced computer vision technology.
            </p>
            <div className="flex gap-4 justify-center">
              <Link href="/dashboard">
                <Button size="lg">
                  Get Started
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/">
                <Button variant="outline" size="lg">Learn More</Button>
              </Link>
            </div>
          </div>
        </section>

        <section className="py-16 bg-background">
          <div className="container mx-auto">
            <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <FileVideo className="h-5 w-5 mr-2" />
                    Fighter Tracking
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Automatically detect and track fighters throughout the match, maintaining consistent identification even during clinches and complex movements.
                  </p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Database className="h-5 w-5 mr-2" />
                    Technique Recognition
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Identify punches, kicks, knees, elbows, and clinches with precise timing. Measure speed, impact, and target areas for comprehensive analysis.
                  </p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <BarChart2 className="h-5 w-5 mr-2" />
                    Match Statistics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">
                    Generate detailed statistics and visualizations including technique distribution, target areas, round activity, and damage assessment.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        <section className="py-16 bg-muted">
          <div className="container mx-auto">
            <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              <div className="text-center">
                <div className="bg-primary text-primary-foreground w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">1</div>
                <h3 className="text-xl font-medium mb-2">Upload Video</h3>
                <p className="text-muted-foreground">
                  Upload your Thai boxing match video in any common format.
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-primary text-primary-foreground w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">2</div>
                <h3 className="text-xl font-medium mb-2">AI Analysis</h3>
                <p className="text-muted-foreground">
                  Our AI engine processes the video, detecting fighters and techniques.
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-primary text-primary-foreground w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">3</div>
                <h3 className="text-xl font-medium mb-2">Generate Statistics</h3>
                <p className="text-muted-foreground">
                  The system creates comprehensive match statistics and visualizations.
                </p>
              </div>
              
              <div className="text-center">
                <div className="bg-primary text-primary-foreground w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4">4</div>
                <h3 className="text-xl font-medium mb-2">Review & Share</h3>
                <p className="text-muted-foreground">
                  Review the analysis, export reports, and share insights with others.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16 bg-background">
          <div className="container mx-auto">
            <div className="max-w-3xl mx-auto text-center">
              <h2 className="text-3xl font-bold mb-6">Ready to analyze your matches?</h2>
              <p className="text-xl text-muted-foreground mb-8">
                Get started with Thai Boxing Vision today and unlock detailed insights into your matches.
              </p>
              <Link href="/dashboard">
                <Button size="lg">
                  Get Started
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="bg-muted py-8">
        <div className="container mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <h2 className="text-xl font-bold">Thai Boxing Vision</h2>
              <p className="text-muted-foreground">AI-powered match analysis</p>
            </div>
            <div className="flex gap-8">
              <div>
                <h3 className="font-medium mb-2">Product</h3>
                <ul className="space-y-1">
                  <li><Link href="/" className="text-muted-foreground hover:text-foreground">Features</Link></li>
                  <li><Link href="/" className="text-muted-foreground hover:text-foreground">Pricing</Link></li>
                  <li><Link href="/" className="text-muted-foreground hover:text-foreground">Documentation</Link></li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium mb-2">Company</h3>
                <ul className="space-y-1">
                  <li><Link href="/" className="text-muted-foreground hover:text-foreground">About</Link></li>
                  <li><Link href="/" className="text-muted-foreground hover:text-foreground">Blog</Link></li>
                  <li><Link href="/" className="text-muted-foreground hover:text-foreground">Contact</Link></li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium mb-2">Legal</h3>
                <ul className="space-y-1">
                  <li><Link href="/" className="text-muted-foreground hover:text-foreground">Privacy</Link></li>
                  <li><Link href="/" className="text-muted-foreground hover:text-foreground">Terms</Link></li>
                </ul>
              </div>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t text-center text-muted-foreground">
            <p>© 2025 Thai Boxing Vision. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
