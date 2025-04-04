import { NextRequest } from 'next/server';

export const runtime = 'edge';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return new Response(JSON.stringify({ error: 'No file provided' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // In a real implementation, we would process the video file here
    // For now, we'll simulate processing and return mock data
    
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Generate mock analysis results
    const mockResults = generateMockResults();
    
    return new Response(JSON.stringify(mockResults), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
    
  } catch (error) {
    console.error('Error processing video:', error);
    return new Response(JSON.stringify({ error: 'Failed to process video' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

function generateMockResults() {
  const techniqueTypes = [
    "JAB", "CROSS", "HOOK", "UPPERCUT", 
    "FRONT_KICK", "ROUNDHOUSE_KICK", "SIDE_KICK",
    "STRAIGHT_KNEE", "DIAGONAL_KNEE",
    "HORIZONTAL_ELBOW", "DIAGONAL_ELBOW"
  ];
  
  const targets = ["HEAD", "BODY", "LEGS", "ARMS"];
  const fighters = ["Red Corner", "Blue Corner"];
  
  const techniques = [];
  
  // Generate 50 random techniques
  for (let i = 0; i < 50; i++) {
    const roundNum = Math.floor(Math.random() * 5) + 1;
    const roundTime = Math.floor(Math.random() * 180);
    const minutes = Math.floor(roundTime / 60);
    const seconds = roundTime % 60;
    const timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    const technique = {
      id: i,
      timestamp: roundNum * 240 - 60 + roundTime, // Convert to video timestamp
      round: roundNum,
      roundTime: timeString,
      fighter: fighters[Math.floor(Math.random() * fighters.length)],
      technique: techniqueTypes[Math.floor(Math.random() * techniqueTypes.length)],
      target: targets[Math.floor(Math.random() * targets.length)],
      speed: Math.floor(Math.random() * 100) + 50,
      impact: (Math.random() * 0.9 + 0.1).toFixed(2)
    };
    
    techniques.push(technique);
  }
  
  // Sort by timestamp
  techniques.sort((a, b) => a.timestamp - b.timestamp);
  
  // Calculate statistics
  const redCornerTechniques = techniques.filter(t => t.fighter === "Red Corner");
  const blueCornerTechniques = techniques.filter(t => t.fighter === "Blue Corner");
  
  // Technique distribution
  const techniqueDistribution = {};
  techniques.forEach(t => {
    if (!techniqueDistribution[t.technique]) {
      techniqueDistribution[t.technique] = 0;
    }
    techniqueDistribution[t.technique]++;
  });
  
  // Target distribution
  const targetDistribution = {};
  techniques.forEach(t => {
    if (!targetDistribution[t.target]) {
      targetDistribution[t.target] = 0;
    }
    targetDistribution[t.target]++;
  });
  
  // Round activity
  const roundActivity = {};
  for (let i = 1; i <= 5; i++) {
    roundActivity[i] = {
      "Red Corner": techniques.filter(t => t.round === i && t.fighter === "Red Corner").length,
      "Blue Corner": techniques.filter(t => t.round === i && t.fighter === "Blue Corner").length
    };
  }
  
  return {
    matchId: `match_${Date.now()}`,
    date: new Date().toISOString().split('T')[0],
    duration: 15 * 60, // 15 minutes in seconds
    rounds: 5,
    techniques,
    statistics: {
      totalTechniques: techniques.length,
      redCornerTechniques: redCornerTechniques.length,
      blueCornerTechniques: blueCornerTechniques.length,
      techniqueDistribution,
      targetDistribution,
      roundActivity
    },
    fighters: {
      "Red Corner": {
        totalTechniques: redCornerTechniques.length,
        avgSpeed: Math.round(redCornerTechniques.reduce((sum, t) => sum + t.speed, 0) / redCornerTechniques.length),
        avgImpact: (redCornerTechniques.reduce((sum, t) => sum + parseFloat(t.impact), 0) / redCornerTechniques.length).toFixed(2),
        mostCommonTechnique: getMostCommon(redCornerTechniques.map(t => t.technique)),
        mostCommonTarget: getMostCommon(redCornerTechniques.map(t => t.target))
      },
      "Blue Corner": {
        totalTechniques: blueCornerTechniques.length,
        avgSpeed: Math.round(blueCornerTechniques.reduce((sum, t) => sum + t.speed, 0) / blueCornerTechniques.length),
        avgImpact: (blueCornerTechniques.reduce((sum, t) => sum + parseFloat(t.impact), 0) / blueCornerTechniques.length).toFixed(2),
        mostCommonTechnique: getMostCommon(blueCornerTechniques.map(t => t.technique)),
        mostCommonTarget: getMostCommon(blueCornerTechniques.map(t => t.target))
      }
    }
  };
}

function getMostCommon(arr) {
  const counts = {};
  let maxItem = null;
  let maxCount = 0;
  
  for (const item of arr) {
    counts[item] = (counts[item] || 0) + 1;
    if (counts[item] > maxCount) {
      maxItem = item;
      maxCount = counts[item];
    }
  }
  
  return maxItem;
}
