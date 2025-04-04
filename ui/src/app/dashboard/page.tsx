"use client";

import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FileVideo, BarChart2, Activity, Clock, Users, Settings, Home } from "lucide-react";
import Link from "next/link";

export default function Dashboard() {
  const [recentMatches] = useState([
    { id: 1, name: "Match 1", date: "Apr 2, 2025", fighters: "Fighter A vs Fighter B", techniques: 124, duration: "15:00" },
    { id: 2, name: "Match 2", date: "Apr 3, 2025", fighters: "Fighter C vs Fighter D", techniques: 98, duration: "15:00" },
    { id: 3, name: "Match 3", date: "Apr 4, 2025", fighters: "Fighter E vs Fighter F", techniques: 112, duration: "15:00" },
  ]);

  return (
    <div className="container mx-auto py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Thai Boxing Vision Dashboard</h1>
        <div className="flex gap-2">
          <Link href="/">
            <Button variant="outline">
              <Home className="h-4 w-4 mr-2" />
              Home
            </Button>
          </Link>
          <Link href="/compare">
            <Button variant="outline">
              <BarChart2 className="h-4 w-4 mr-2" />
              Compare Matches
            </Button>
          </Link>
          <Button variant="outline">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center">
              <FileVideo className="h-5 w-5 mr-2" />
              Quick Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <Label htmlFor="quick-analysis-file">Upload Match Video</Label>
                <div className="flex mt-1">
                  <Input id="quick-analysis-file" type="file" accept="video/*" />
                </div>
              </div>
              <Button className="w-full">Start Analysis</Button>
              <p className="text-xs text-muted-foreground">
                Upload a Thai boxing match video for quick analysis of techniques, strikes, and match statistics.
              </p>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center">
              <BarChart2 className="h-5 w-5 mr-2" />
              Statistics Overview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Matches</span>
                <span className="font-medium">3</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Techniques</span>
                <span className="font-medium">334</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg. Techniques/Match</span>
                <span className="font-medium">111</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Most Common Technique</span>
                <span className="font-medium">Jab (24%)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Most Common Target</span>
                <span className="font-medium">Head (42%)</span>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg flex items-center">
              <Users className="h-5 w-5 mr-2" />
              Fighter Database
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Fighters</span>
                <span className="font-medium">6</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Most Active</span>
                <span className="font-medium">Fighter A (2 matches)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Highest Strike Rate</span>
                <span className="font-medium">Fighter C (42/round)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Fastest Strikes</span>
                <span className="font-medium">Fighter E (187 km/h)</span>
              </div>
            </div>
            <Button variant="outline" className="w-full mt-4">View All Fighters</Button>
          </CardContent>
        </Card>
      </div>
      
      <Tabs defaultValue="recent" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="recent">
            <Clock className="h-4 w-4 mr-2" />
            Recent Matches
          </TabsTrigger>
          <TabsTrigger value="techniques">
            <Activity className="h-4 w-4 mr-2" />
            Technique Analysis
          </TabsTrigger>
          <TabsTrigger value="fighters">
            <Users className="h-4 w-4 mr-2" />
            Fighter Stats
          </TabsTrigger>
          <TabsTrigger value="trends">
            <BarChart2 className="h-4 w-4 mr-2" />
            Trends
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="recent" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Recent Matches</CardTitle>
              <CardDescription>
                View and analyze your recently processed matches
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-3 px-4">Match Name</th>
                      <th className="text-left py-3 px-4">Date</th>
                      <th className="text-left py-3 px-4">Fighters</th>
                      <th className="text-left py-3 px-4">Techniques</th>
                      <th className="text-left py-3 px-4">Duration</th>
                      <th className="text-left py-3 px-4">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentMatches.map((match) => (
                      <tr key={match.id} className="border-b hover:bg-muted/50">
                        <td className="py-3 px-4">{match.name}</td>
                        <td className="py-3 px-4">{match.date}</td>
                        <td className="py-3 px-4">{match.fighters}</td>
                        <td className="py-3 px-4">{match.techniques}</td>
                        <td className="py-3 px-4">{match.duration}</td>
                        <td className="py-3 px-4">
                          <div className="flex gap-2">
                            <Button variant="outline" size="sm">View</Button>
                            <Button variant="outline" size="sm">Report</Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="techniques" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Technique Analysis</CardTitle>
              <CardDescription>
                Breakdown of techniques across all analyzed matches
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Technique Distribution</h3>
                  <div className="aspect-square bg-muted rounded-md flex items-center justify-center">
                    <div className="text-center p-12">
                      <BarChart2 className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
                      <p className="text-muted-foreground">
                        Technique distribution chart will appear here
                      </p>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-4">Target Areas</h3>
                  <div className="aspect-square bg-muted rounded-md flex items-center justify-center">
                    <div className="text-center p-12">
                      <Activity className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
                      <p className="text-muted-foreground">
                        Target area distribution chart will appear here
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="md:col-span-2">
                  <h3 className="text-lg font-medium mb-4">Technique Breakdown</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-3 px-4">Technique</th>
                          <th className="text-left py-3 px-4">Count</th>
                          <th className="text-left py-3 px-4">Percentage</th>
                          <th className="text-left py-3 px-4">Avg. Speed</th>
                          <th className="text-left py-3 px-4">Avg. Impact</th>
                          <th className="text-left py-3 px-4">Success Rate</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b hover:bg-muted/50">
                          <td className="py-3 px-4">Jab</td>
                          <td className="py-3 px-4">78</td>
                          <td className="py-3 px-4">23.4%</td>
                          <td className="py-3 px-4">142 km/h</td>
                          <td className="py-3 px-4">0.65</td>
                          <td className="py-3 px-4">82%</td>
                        </tr>
                        <tr className="border-b hover:bg-muted/50">
                          <td className="py-3 px-4">Cross</td>
                          <td className="py-3 px-4">62</td>
                          <td className="py-3 px-4">18.6%</td>
                          <td className="py-3 px-4">156 km/h</td>
                          <td className="py-3 px-4">0.78</td>
                          <td className="py-3 px-4">75%</td>
                        </tr>
                        <tr className="border-b hover:bg-muted/50">
                          <td className="py-3 px-4">Roundhouse Kick</td>
                          <td className="py-3 px-4">54</td>
                          <td className="py-3 px-4">16.2%</td>
                          <td className="py-3 px-4">132 km/h</td>
                          <td className="py-3 px-4">0.85</td>
                          <td className="py-3 px-4">68%</td>
                        </tr>
                        <tr className="border-b hover:bg-muted/50">
                          <td className="py-3 px-4">Straight Knee</td>
                          <td className="py-3 px-4">42</td>
                          <td className="py-3 px-4">12.6%</td>
                          <td className="py-3 px-4">128 km/h</td>
                          <td className="py-3 px-4">0.82</td>
                          <td className="py-3 px-4">72%</td>
                        </tr>
                        <tr className="border-b hover:bg-muted/50">
                          <td className="py-3 px-4">Horizontal Elbow</td>
                          <td className="py-3 px-4">38</td>
                          <td className="py-3 px-4">11.4%</td>
                          <td className="py-3 px-4">118 km/h</td>
                          <td className="py-3 px-4">0.76</td>
                          <td className="py-3 px-4">70%</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="fighters" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Fighter Statistics</CardTitle>
              <CardDescription>
                Performance metrics for all fighters
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-3 px-4">Fighter</th>
                      <th className="text-left py-3 px-4">Matches</th>
                      <th className="text-left py-3 px-4">Win/Loss</th>
                      <th className="text-left py-3 px-4">Total Techniques</th>
                      <th className="text-left py-3 px-4">Favorite Technique</th>
                      <th className="text-left py-3 px-4">Avg. Speed</th>
                      <th className="text-left py-3 px-4">Avg. Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="py-3 px-4">Fighter A</td>
                      <td className="py-3 px-4">2</td>
                      <td className="py-3 px-4">1/1</td>
                      <td className="py-3 px-4">68</td>
                      <td className="py-3 px-4">Jab (32%)</td>
                      <td className="py-3 px-4">138 km/h</td>
                      <td className="py-3 px-4">0.72</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="py-3 px-4">Fighter B</td>
                      <td className="py-3 px-4">1</td>
                      <td className="py-3 px-4">0/1</td>
                      <td className="py-3 px-4">56</td>
                      <td className="py-3 px-4">Cross (28%)</td>
                      <td className="py-3 px-4">145 km/h</td>
                      <td className="py-3 px-4">0.68</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="py-3 px-4">Fighter C</td>
                      <td className="py-3 px-4">1</td>
                      <td className="py-3 px-4">1/0</td>
                      <td className="py-3 px-4">52</td>
                      <td className="py-3 px-4">Roundhouse Kick (24%)</td>
                      <td className="py-3 px-4">132 km/h</td>
                      <td className="py-3 px-4">0.75</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="py-3 px-4">Fighter D</td>
                      <td className="py-3 px-4">1</td>
                      <td className="py-3 px-4">0/1</td>
                      <td className="py-3 px-4">46</td>
                      <td className="py-3 px-4">Jab (30%)</td>
                      <td className="py-3 px-4">136 km/h</td>
                      <td className="py-3 px-4">0.70</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="py-3 px-4">Fighter E</td>
                      <td className="py-3 px-4">1</td>
                      <td className="py-3 px-4">1/0</td>
                      <td className="py-3 px-4">62</td>
                      <td className="py-3 px-4">Straight Knee (22%)</td>
                      <td className="py-3 px-4">152 km/h</td>
                      <td className="py-3 px-4">0.82</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="py-3 px-4">Fighter F</td>
                      <td className="py-3 px-4">1</td>
                      <td className="py-3 px-4">0/1</td>
                      <td className="py-3 px-4">50</td>
                      <td className="py-3 px-4">Horizontal Elbow (26%)</td>
                      <td className="py-3 px-4">128 km/h</td>
                      <td className="py-3 px-4">0.78</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="trends" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Trend Analysis</CardTitle>
              <CardDescription>
                Identify patterns and trends across multiple matches
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-[2/1] bg-muted rounded-md flex items-center justify-center">
                <div className="text-center p-12">
                  <BarChart2 className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">
                    Analyze more matches to view trend data and insights
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
