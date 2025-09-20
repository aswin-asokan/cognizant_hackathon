import React, { useState, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui1/card';
import { Button } from '../components/ui1/button';
import { Input } from '../components/ui1/input';
import { Badge } from '../components/ui1/badge';
import { Avatar, AvatarFallback } from '../components/ui1/avatar';
import { Progress } from '../components/ui1/progress';
import { Alert, AlertDescription } from '../components/ui1/alert';
import { 
  BarChart3, 
  History, 
  AlertTriangle, 
  Search, 
  Bell, 
  Menu,
  X,
  FileUp,
  CheckCircle,
  XCircle,
  Shield,
  Package,
  Activity,
  LogOut
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const sidebarItems = [
  { icon: BarChart3, label: 'Dashboard', active: true },
  { icon: History, label: 'History', active: false }
];

const FraudDetectionChart = ({ normalCount, fraudulentCount }: { normalCount: number; fraudulentCount: number }) => {
  const total = normalCount + fraudulentCount;
  const normalPercentage = ((normalCount / total) * 100).toFixed(1);
  const fraudulentPercentage = ((fraudulentCount / total) * 100).toFixed(1);
  
  const data = [
    { name: 'Normal', value: normalCount, color: '#10B981', percentage: normalPercentage },
    { name: 'Fraudulent', value: fraudulentCount, color: '#EF4444', percentage: fraudulentPercentage }
  ];

  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <defs>
            <filter id="shadow">
              <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.1"/>
            </filter>
          </defs>
          <Pie 
            data={data} 
            cx="50%" 
            cy="50%" 
            innerRadius={80} 
            outerRadius={120} 
            startAngle={90} 
            endAngle={450}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} filter="url(#shadow)" />
            ))}
          </Pie>
          <Tooltip 
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#F9FAFB'
            }}
            formatter={(value: any, name: string) => [
              `${value} shipments (${data.find(d => d.name === name)?.percentage}%)`, 
              name
            ]}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="text-3xl font-semibold text-gray-900">{total}</div>
        <div className="text-sm text-gray-500">Total Shipments</div>
      </div>
      
      <div className="flex justify-center mt-6 space-x-8">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-green-500 rounded-full"></div>
          <span className="text-gray-700">Normal ({normalPercentage}%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-red-500 rounded-full"></div>
          <span className="text-gray-700">Fraudulent ({fraudulentPercentage}%)</span>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [predictionResult, setPredictionResult] = useState<{
    normalCount: number;
    fraudulentCount: number;
    fileName: string;
    anomalyScore?: number;
    processingTime?: string;
    processedCsv?: string;
  } | null>(null);
  const [errorMessage, setErrorMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_BASE_URL = 'http://localhost:5000';

  const processFileWithIsolationTest = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/analyze-fraud`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  };

  const downloadProcessedCSV = (csvString: string, filename: string) => {
    const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleFileUpload = async (file: File) => {
    setUploadStatus('uploading');
    setUploadProgress(0);
    setErrorMessage('');

    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) { clearInterval(progressInterval); return 90; }
        return prev + 20;
      });
    }, 500);

    try {
      const result = await processFileWithIsolationTest(file);
      clearInterval(progressInterval);
      setUploadProgress(100);
      setUploadStatus('success');

      setPredictionResult({
        normalCount: result.normal_count || 0,
        fraudulentCount: result.fraudulent_count || 0,
        fileName: file.name,
        anomalyScore: result.anomaly_score,
        processingTime: result.processing_time,
        processedCsv: result.processed_csv
      });

    } catch (error) {
      clearInterval(progressInterval);
      setUploadStatus('error');
      setUploadProgress(0);
      setErrorMessage(error instanceof Error ? error.message : 'Failed to process file.');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === 'text/csv') handleFileUpload(files[0]);
    else setErrorMessage('Please upload a valid CSV file');
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) handleFileUpload(files[0]);
  };

  const resetUpload = () => {
    setUploadStatus('idle');
    setUploadProgress(0);
    setPredictionResult(null);
    setErrorMessage('');
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-gray-200 shadow-lg transform ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 ease-in-out lg:inset-0 flex flex-col`}>
        <div className="flex items-center justify-between h-16 px-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gray-900 rounded-lg flex items-center justify-center">
              <Shield className="h-5 w-5 text-white" />
            </div>
            <span className="text-gray-900 font-semibold">FraudGuard</span>
          </div>
          <Button variant="ghost" size="sm" className="lg:hidden text-gray-400 hover:text-gray-600" onClick={() => setSidebarOpen(false)}>
            <X className="h-5 w-5" />
          </Button>
        </div>
        
        <nav className="mt-8 px-4 flex flex-col h-full">
          <ul className="space-y-2">
            {sidebarItems.map((item, index) => (
              <li key={index}>
                <Button variant="ghost" className={`w-full justify-start text-left ${item.active ? 'bg-gray-900 text-white hover:bg-gray-800' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'}`}>
                  <item.icon className="h-5 w-5 mr-3" />
                  {item.label}
                </Button>
              </li>
            ))}
          </ul>
          <div className="mt-auto mb-6">
            <Button variant="ghost" className="w-full justify-start text-left text-gray-600 hover:text-red-600 hover:bg-red-50">
              <LogOut className="h-5 w-5 mr-3" />
              Logout
            </Button>
          </div>
        </nav>
      </div>

      <div className="lg:ml-64">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 h-16 shadow-sm">
          <div className="flex items-center justify-between h-full px-6">
            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="sm" className="lg:hidden text-gray-400 hover:text-gray-600" onClick={() => setSidebarOpen(true)}>
                <Menu className="h-5 w-5" />
              </Button>
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gray-900 rounded-lg flex items-center justify-center">
                  <Shield className="h-5 w-5 text-white" />
                </div>
                <span className="text-gray-900 font-semibold">FraudGuard</span>
              </div>
            </div>
            <div className="flex-1 max-w-md mx-8">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
                <Input placeholder="Search shipments..." className="pl-10 bg-gray-50 border-gray-200 focus:border-gray-400"/>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="sm" className="text-gray-400 hover:text-gray-600">
                <Bell className="h-5 w-5" />
              </Button>
              <Avatar>
                <AvatarFallback className="bg-gray-900 text-white">{'U'}</AvatarFallback>
              </Avatar>
            </div>
          </div>
        </header>

        <main className="p-6 space-y-8">
          <div>
            <h1 className="text-3xl font-semibold text-gray-900">SHIPGUARD</h1>
            <p className="text-gray-600 mt-2">Upload shipment data and view AI-powered fraud detection results</p>
          </div>

          {errorMessage && (
            <Alert className="border-red-200 bg-red-50">
              <AlertTriangle className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-800">{errorMessage}</AlertDescription>
            </Alert>
          )}

          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="bg-white shadow-lg border border-gray-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Total Shipments</p>
                    <p className="text-2xl font-semibold text-gray-900 mt-1">
                      {predictionResult ? (predictionResult.normalCount + predictionResult.fraudulentCount).toLocaleString() : '0'}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                    <Package className="h-6 w-6 text-purple-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white shadow-lg border border-gray-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Detection Rate</p>
                    <p className="text-2xl font-semibold text-gray-900 mt-1">
                      {predictionResult ? `${(((predictionResult.fraudulentCount) / (predictionResult.normalCount + predictionResult.fraudulentCount)) * 100).toFixed(1)}%` : '0%'}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
                    <AlertTriangle className="h-6 w-6 text-red-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white shadow-lg border border-gray-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Processing Time</p>
                    <p className="text-2xl font-semibold text-gray-900 mt-1">
                      {predictionResult?.processingTime || '0s'}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                    <Activity className="h-6 w-6 text-blue-600" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Upload Section */}
            <Card className="bg-white shadow-lg border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-900 flex items-center">
                  <FileUp className="h-6 w-6 mr-3 text-gray-700" />
                  Upload Shipment Data
                </CardTitle>
                <CardDescription className="text-gray-600">
                  Upload a CSV file for fraud analysis using Isolation Forest
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${uploadStatus === 'error' ? 'border-red-300 bg-red-50' : uploadStatus === 'success' ? 'border-green-300 bg-green-50' : 'border-gray-300 hover:border-gray-400 bg-gray-50 hover:bg-gray-100'}`}
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                >
                  <div className="flex flex-col items-center space-y-4">
                    {uploadStatus === 'idle' && (
                      <>
                        <FileUp className="h-12 w-12 text-gray-400" />
                        <div>
                          <p className="text-gray-900 font-medium">Drop your CSV file here</p>
                          <p className="text-gray-500">or click to browse files</p>
                        </div>
                        <Button variant="outline" onClick={() => fileInputRef.current?.click()} className="border-gray-300 text-gray-700 hover:bg-gray-100">
                          Browse Files
                        </Button>
                      </>
                    )}

                    {uploadStatus === 'uploading' && (
                      <>
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
                        <div className="w-full max-w-sm">
                          <p className="text-gray-900 font-medium mb-3">Processing with Isolation Forest...</p>
                          <Progress value={uploadProgress} className="h-3" />
                          <p className="text-gray-500 text-sm mt-2">{uploadProgress}% complete</p>
                        </div>
                      </>
                    )}

                    {uploadStatus === 'success' && (
                      <>
                        <CheckCircle className="h-12 w-12 text-green-500" />
                        <div>
                          <p className="text-gray-900 font-medium">Analysis complete!</p>
                          <p className="text-gray-500">Fraud detection results ready</p>
                        </div>
                        <Button variant="outline" onClick={resetUpload} className="border-gray-300 text-gray-700 hover:bg-gray-100">
                          Upload New File
                        </Button>
                      </>
                    )}

                    {uploadStatus === 'error' && (
                      <>
                        <XCircle className="h-12 w-12 text-red-500" />
                        <div>
                          <p className="text-gray-900 font-medium">Analysis failed</p>
                          <p className="text-red-500">{errorMessage || 'Please check your file and try again'}</p>
                        </div>
                        <Button variant="outline" onClick={resetUpload} className="border-gray-300 text-gray-700 hover:bg-gray-100">
                          Try Again
                        </Button>
                      </>
                    )}
                  </div>
                </div>

                <input ref={fileInputRef} type="file" accept=".csv" onChange={handleFileInputChange} className="hidden" />
              </CardContent>
            </Card>

            {/* Fraud Prediction Result */}
            <Card className="bg-white shadow-lg border border-gray-200">
              <CardHeader>
                <CardTitle className="text-gray-900 flex items-center">
                  <Shield className="h-6 w-6 mr-3 text-gray-700" />
                  Isolation Forest Results
                </CardTitle>
                <CardDescription className="text-gray-600">
                  {predictionResult ? `Results for ${predictionResult.fileName}` : 'Upload data to view fraud detection results'}
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col items-center justify-center">
                {predictionResult ? (
                  <div className="w-full">
                    <FraudDetectionChart 
                      normalCount={predictionResult.normalCount}
                      fraudulentCount={predictionResult.fraudulentCount}
                    />

                    {/* Summary Stats */}
                    <div className="grid grid-cols-2 gap-4 mt-8">
                      <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                        <div className="flex items-center space-x-2">
                          <CheckCircle className="h-5 w-5 text-green-600" />
                          <span className="text-green-800 font-medium">Normal</span>
                        </div>
                        <p className="text-2xl font-semibold text-green-900 mt-1">{predictionResult.normalCount}</p>
                        <p className="text-green-700 text-sm">shipments detected</p>
                      </div>
                      
                      <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                        <div className="flex items-center space-x-2">
                          <AlertTriangle className="h-5 w-5 text-red-600" />
                          <span className="text-red-800 font-medium">Anomalies</span>
                        </div>
                        <p className="text-2xl font-semibold text-red-900 mt-1">{predictionResult.fraudulentCount}</p>
                        <p className="text-red-700 text-sm">outliers detected</p>
                      </div>
                    </div>

                    {predictionResult.anomalyScore && (
                      <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                        <p className="text-blue-800 text-sm">
                          Average Anomaly Score: <span className="font-medium">{predictionResult.anomalyScore.toFixed(3)}</span>
                        </p>
                      </div>
                    )}

                    {predictionResult.processedCsv && (
                      <div className="mt-6 flex justify-center">
                        <Button
                          variant="outline"
                          onClick={() => downloadProcessedCSV(predictionResult.processedCsv!, `processed_${predictionResult.fileName}`)}
                        >
                          Download Processed CSV
                        </Button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Shield className="h-8 w-8 text-gray-400" />
                    </div>
                    <p className="text-gray-500">No data uploaded yet</p>
                    <p className="text-gray-400 text-sm mt-1">Upload a CSV file to see analysis results</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </main>
      </div>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        ></div>
      )}
    </div>
  );
}

