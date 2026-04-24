import { useState } from "react";
import { Button } from "@/app/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/app/components/ui/select";
import { Card } from "@/app/components/ui/card";
import { Play } from "lucide-react";

// Malaysia states and districts data based on dataset
const MALAYSIA_DATA: Record<string, string[]> = {
  "Johor": ["Batu Pahat", "Johor Bahru", "Kluang", "Kota Tinggi", "Kulai", "Mersing", "Muar", "Pontian", "Segamat", "Tangkak"],
  "Kedah": ["Baling", "Bandar Baharu", "Kota Setar", "Kuala Muda", "Kubang Pasu", "Kulim", "Langkawi", "Padang Terap", "Pendang", "Pokok Sena", "Sik", "Yan"],
  "Kelantan": ["Bachok", "Gua Musang", "Jeli", "Kecil Lojing", "Kota Bharu", "Kuala Krai", "Machang", "Pasir Mas", "Pasir Puteh", "Tanah Merah", "Tumpat"],
  "Melaka": ["Alor Gajah", "Jasin", "Melaka Tengah"],
  "Negeri Sembilan": ["Jelebu", "Jempol", "Kuala Pilah", "Port Dickson", "Rembau", "Seremban", "Tampin"],
  "Pahang": ["Bentong", "Bera", "Cameron Highlands", "Jerantut", "Kuantan", "Lipis", "Maran", "Pekan", "Raub", "Rompin", "Temerloh"],
  "Perak": ["Bagan Datuk", "Batang Padang", "Hilir Perak", "Hulu Perak", "Kampar", "Kerian", "Kinta", "Kuala Kangsar", "Larut dan Matang", "Manjung", "Muallim", "Perak Tengah", "Selama"],
  "Perlis": ["Perlis"],
  "Pulau Pinang": ["Barat Daya", "Seberang Perai Selatan", "Seberang Perai Tengah", "Seberang Perai Utara", "Timur Laut"],
  "Sabah": ["Beaufort", "Beluran", "Kalabakan", "Keningau", "Kinabatangan", "Kota Belud", "Kota Kinabalu", "Kota Marudu", "Kuala Penyu", "Kudat", "Kunak", "Lahad Datu", "Nabawan", "Papar", "Penampang", "Pitas", "Putatan", "Ranau", "Sandakan", "Semporna", "Sipitang", "Tambunan", "Tawau", "Telupid", "Tenom", "Tongod", "Tuaran"],
  "Sarawak": ["Asajaya", "Bau", "Belaga", "Beluru", "Betong", "Bintulu", "Bukit Mabong", "Dalat", "Daro", "Julau", "Kabong", "Kanowit", "Kapit", "Kuching", "Lawas", "Limbang", "Lubok Antu", "Lundu", "Maradong", "Marudi", "Matu", "Miri", "Mukah", "Pakan", "Pusa", "Samarahan", "Saratok", "Sarikei", "Sebauh", "Selangau", "Serian", "Sibu", "Simunjan", "Song", "Sri Aman", "Subis", "Tanjung Manis", "Tatau", "Tebedu", "Telang Usan"],
  "Selangor": ["Gombak", "Klang", "Kuala Langat", "Kuala Selangor", "Petaling", "Sabak Bernam", "Sepang", "Ulu Langat", "Ulu Selangor"],
  "Terengganu": ["Besut", "Dungun", "Hulu Terengganu", "Kemaman", "Kuala Nerus", "Kuala Terengganu", "Marang", "Setiu"],
  "W.P. Kuala Lumpur": ["W.P. Kuala Lumpur"],
  "W.P. Labuan": ["W.P. Labuan"],
  "W.P. Putrajaya": ["W.P. Putrajaya"],
};

const STATES = Object.keys(MALAYSIA_DATA);

export function AnalysisConfiguration() {
  const [geoLevel, setGeoLevel] = useState<string>("state");
  const [selectedState, setSelectedState] = useState<string>("");
  const [selectedDistrict, setSelectedDistrict] = useState<string>("");

  const handleGeoLevelChange = (value: string) => {
    setGeoLevel(value);
    setSelectedState("");
    setSelectedDistrict("");
  };

  const handleStateChange = (value: string) => {
    setSelectedState(value);
    setSelectedDistrict("");
  };

  return (
    <Card className="p-8 bg-card border-border">
      <h2 className="mb-8 text-foreground tracking-tight">
        Analysis Configuration
      </h2>

      <div className="space-y-6">
        {/* Geographic Level */}
        <div className="space-y-3">
          <label className="text-foreground/90 block">
            Geographic Level
          </label>
          <Select value={geoLevel} onValueChange={handleGeoLevelChange}>
            <SelectTrigger className="w-full bg-input-background border-border">
              <SelectValue placeholder="Select geographic level" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="state">State</SelectItem>
              <SelectItem value="district">District</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* State - always visible */}
        <div className="space-y-3">
          <label className="text-foreground/90 block">
            State
          </label>
          <Select value={selectedState} onValueChange={handleStateChange}>
            <SelectTrigger className="w-full bg-input-background border-border">
              <SelectValue placeholder="Select state" />
            </SelectTrigger>
            <SelectContent>
              {STATES.map((state) => (
                <SelectItem key={state} value={state}>
                  {state}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* District - only visible when geographic level is "district" */}
        {geoLevel === "district" && (
          <div className="space-y-3">
            <label className="text-foreground/90 block">
              District
            </label>
            <Select 
              value={selectedDistrict} 
              onValueChange={setSelectedDistrict}
              disabled={!selectedState}
            >
              <SelectTrigger className="w-full bg-input-background border-border">
                <SelectValue placeholder={selectedState ? "Select district" : "Select a state first"} />
              </SelectTrigger>
              <SelectContent>
                {selectedState && MALAYSIA_DATA[selectedState]?.map((district) => (
                  <SelectItem key={district} value={district}>
                    {district}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {/* Year */}
        <div className="space-y-3">
          <label className="text-foreground/90 block">
            Year
          </label>
          <Select defaultValue="2022">
            <SelectTrigger className="w-full bg-input-background border-border">
              <SelectValue placeholder="Select year" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="2022">2022</SelectItem>
              <SelectItem value="2019">2019</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Modelling Type */}
        <div className="space-y-3">
          <label className="text-foreground/90 block">
            Modelling Type
          </label>
          <Select defaultValue="multiple-linear-regression">
            <SelectTrigger className="w-full bg-input-background border-border">
              <SelectValue placeholder="Select modelling type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="multiple-linear-regression">
                Multiple Linear Regression Analysis
              </SelectItem>
              <SelectItem value="ridge-regression">
                Ridge Regression
              </SelectItem>
              <SelectItem value="random-forest-regressor">
                Random Forest Regressor
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="pt-4">
          <Button
            className="w-full bg-primary hover:bg-primary/90 text-primary-foreground py-6"
            size="lg"
          >
            <Play className="mr-2 h-5 w-5" />
            Execute Analysis (Prototype)
          </Button>
        </div>
      </div>
    </Card>
  );
}