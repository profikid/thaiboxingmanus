"use client";

import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from "next/link";

export default function Compare() {
  const [selectedMatches, setSelectedMatches] = useState({
    match1: "",
    match2: ""
  });

  return (
    <div className="container mx-auto py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Match Comparison</h1>
        <Link href="/">
          <Button variant="outline">Back to Dashboard</Button>
        </Link>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle>Match 1</CardTitle>
            <CardDescription>
              Select the first match for comparison
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <Label htmlFor="match1-file">Upload Match Video</Label>
                <div className="flex mt-1">
                  <Input id="match1-file" type="file" accept="video/*" />
                  <Button className="ml-2">Upload</Button>
                </div>
              </div>
              
              <div>
                <Label htmlFor="match1-select">Or Select Existing Match</Label>
                <select 
                  id="match1-select" 
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 mt-1"
                  value={selectedMatches.match1}
                  onChange={(e) => setSelectedMatches({...selectedMatches, match1: e.target.value})}
                >
                  <option value="">Select a match...</option>
                  <option value="match1">Match 1 - Apr 2, 2025</option>
                  <option value="match2">Match 2 - Apr 3, 2025</option>
                  <option value="match3">Match 3 - Apr 4, 2025</option>
                </select>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Match 2</CardTitle>
            <CardDescription>
              Select the second match for comparison
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <Label htmlFor="match2-file">Upload Match Video</Label>
                <div className="flex mt-1">
                  <Input id="match2-file" type="file" accept="video/*" />
                  <Button className="ml-2">Upload</Button>
                </div>
              </div>
              
              <div>
                <Label htmlFor="match2-select">Or Select Existing Match</Label>
                <select 
                  id="match2-select" 
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 mt-1"
                  value={selectedMatches.match2}
                  onChange={(e) => setSelectedMatches({...selectedMatches, match2: e.target.value})}
                >
                  <option value="">Select a match...</option>
                  <option value="match1">Match 1 - Apr 2, 2025</option>
                  <option value="match2">Match 2 - Apr 3, 2025</option>
                  <option value="match3">Match 3 - Apr 4, 2025</option>
                </select>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Rest of the component remains the same */}
    </div>
  );
}
