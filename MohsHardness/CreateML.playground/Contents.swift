import Cocoa
import CoreML


let modelUrl = URL(fileURLWithPath: "/Users/fx7707/Documents/GitHub/KaggleNotebook/MohsHardness/Hardness pre.mlmodel")




do {
    let compiledUrl = try MLModel.compileModel(at: modelUrl)
    let model = try MLModel(contentsOf: compiledUrl)
//    print("Model compiled \(model.modelDescription)")
//    print(model.modelDescription.inputDescriptionsByName)
    if let file: FileHandle = FileHandle(forReadingAtPath:  "/Users/fx7707/Documents/GitHub/KaggleNotebook/MohsHardness/processedData/processed_test_data.csv"){
        var data = ""
        do {
            data = String(data: try file.readToEnd()!, encoding: String.Encoding.utf8)!
//            print(data)
            var rows = data.components(separatedBy: "\n")
            print(rows[0])
            rows.removeFirst()
            var csvString = "\("id"),\("Hardness")\n"
            for row in rows {
                var columns = row.components(separatedBy: ",")
                
//                print(columns, columns.count)
                if columns.count != 1{
//                    print(columns, columns.count)
                    let r = try MLDictionaryFeatureProvider(dictionary: ["ionenergy_Average": Double(columns[0]),
                                                                         "zaratio_Average": Double(columns[1]),
                                                                         "density_Total": Double(columns[2]),
                                                                         "allelectrons_Total": Double(columns[3]),
                                                                         "allelectrons_Average": Double(columns[4]),
                                                                         "density_Average": Double(columns[5]),
                                                                         "R_vdw_element_Average": Double(columns[6]),
                                                                         "R_cov_element_Average": Double(columns[7]),
                                                                         "atomicweight_Average": Double(columns[8]),
                                                                         "el_neg_chi_Average": Double(columns[9]),
                                                                         "val_e_Average": Double(columns[10])])
                    let output = try model.prediction(from: r)
                    if let pred = output.featureValue(for: "Hardness")?.doubleValue{
                        
                        let id = columns[11]
                        csvString = csvString.appending("\(id), \(pred)\n")
                    }else{
                        print("error")
                        break
                    }
                    
                }
                //check that we have enough columns
                
            }
            let fileManager = FileManager.default

            do {

            let path = try fileManager.url(for: .documentDirectory, in: .allDomainsMask, appropriateFor: nil , create: false )

            let fileURL = path.appendingPathComponent("playgroundResult.csv")
            print(csvString)
            try csvString.write(to: fileURL, atomically: true , encoding: .utf8)
            } catch {

                print("error creating file")

            }
        } catch {
            print(error)
        }
        
    }
    //model.prediction(from: MLFeatureProvider) //Problem
    //It should be like this
    //guard let prediction = try? model.prediction(image: pixelBuffer!) else {
    //    return
    //}
} catch {
    print("Error while compiling \(error.localizedDescription)")
}
