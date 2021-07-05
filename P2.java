package uk.ac.soton.ecs.kk8g18;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class P2 {
	
	public static int CLUSTERS = 500;
    //public static int IMAGES_FOR_VOCABULARY = 10;

    // Patch parameters
	public static float STEP = 8;
	public static float PATCH_SIZE = 12;
	private static LiblinearAnnotator<FImage, String> ann;
	
	
	//LiblinearAnnotator l = new LiblinearAnnotator(null, null, null, 0, 0);
	public static void test() throws FileSystemException {
		
		System.out.println("1");
		GroupedDataset<String, VFSListDataset<FImage>, FImage> allData = new VFSGroupDataset<FImage>("C:\\Users\\karth\\Desktop\\training\\training",ImageUtilities.FIMAGE_READER);
		GroupedRandomSplitter<String, FImage> t =  new GroupedRandomSplitter<String, FImage>(allData, 10, 0, 10);
		
		
		List<float[]> allkeys = new ArrayList<float[]>();

        // extract patches    
		System.out.println("2");
		for (FImage image : t.getTrainingDataset()) {
			List<float[]> sampleList = extractFeature1(image);
			//System.err.println(sampleList.size());

			for(float[] f : sampleList){
				allkeys.add(f);
			}
		}
		System.out.println("3");
		
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
		System.out.println("3.1");
        float[][] data = allkeys.toArray(new float[][]{});
        FloatCentroidsResult result = km.cluster(data);
        System.out.println("3.2");
        HardAssigner<float[], float[], IntFloatPair> assigner = result.defaultHardAssigner();
        System.out.println("3.3");
        
        FeatureExtractor<DoubleFV, FImage> extractor = new PatchClusterFeatureExtractor(assigner);
        System.out.println("4");
        // Create and train a linear classifier.
		System.err.println("Start training...");
		ann = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(t.getTrainingDataset()); 
		
		
		System.out.println(ann.getAnnotations());
        System.err.println("Training finished.");
        
//        for(int i = 0; i<t.getTestDataset().size();i++) {
//        	ann.classify(t.getTestDataset());
//        }
        double co = 0;
        double to = 0;
        
        for(Entry<String, ListDataset<FImage>> images : t.getTestDataset().entrySet()) {
        	for(FImage im: images.getValue()) {
        		ClassificationResult<String> st = ann.classify(im);
        		//System.out.println(st);
        		if(st.getPredictedClasses().contains(images.getKey())) {
        			
        			System.out.println(st.getPredictedClasses());
        			co++;
        		}
        		to++;
        	}
        }
        
        System.out.println(co);
		System.out.println(to);
		double correct = (co/to)*100;
		System.out.println(correct+"%");
		
		
		VFSListDataset<FImage> testing = new VFSListDataset<FImage>("C:\\Users\\karth\\Desktop\\testing",ImageUtilities.FIMAGE_READER);
		Map<String,String> ma = new HashMap<String,String>();
		
		for(int j = 0;j<testing.size();j++) {
			ClassificationResult<String> s = ann.classify(testing.get(j));
			ma.put(testing.getID(j), s.getPredictedClasses().toArray(new String [] {})[0]);
		}
		
		File file = new File("C:\\Users\\karth\\Desktop\\t2.txt");
		BufferedWriter bf = null;;
        
        try{
            
            //create new BufferedWriter for the output file
            bf = new BufferedWriter( new FileWriter(file) );
 
            //iterate map entries
            for(Map.Entry<String, String> entry : ma.entrySet()){
                
                //put key and value separated by a colon
                bf.write( entry.getKey() + " " + entry.getValue() );
                
                //new line
                bf.newLine();
            }
            
            bf.flush();
 
        }catch(IOException e){
            e.printStackTrace();
        }finally{
            
            try{
                //always close the writer
                bf.close();
            }catch(Exception e){}
        }
		
		
	}
	
	
	
	public static List<float[]> extractFeature1(FImage image) {
		
//		List<LocalFeature<SpatialLocation, FloatFV>> areaList = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();
//
//        // Create patch positions
//        RectangleSampler rect = new RectangleSampler(image, STEP, STEP, PATCH_SIZE, PATCH_SIZE);
//
//        // Extract feature from position r.
//        for(Rectangle r : rect){
//            FImage area = image.extractROI(r);
//
//            //2D array to 1D array
//            float[] vector = ArrayUtils.reshape(area.pixels);
//            FloatFV featureV = new FloatFV(vector);
//            //Location of rectangle is location of feature
//            SpatialLocation sl = new SpatialLocation(r.x, r.y);
//            
//            //Generate as a local feature for compatibility with other modules
//            LocalFeature<SpatialLocation, FloatFV> lf = new LocalFeatureImpl<SpatialLocation, FloatFV>(sl,featureV);
//
//            areaList.add(lf);
//        }
//
//        return areaList;	
		
		
		List<float[]> areaList = new ArrayList<float[]>();
		
		        // Create patch positions
		RectangleSampler rect = new RectangleSampler(image, STEP, STEP, PATCH_SIZE, PATCH_SIZE);
		
		        // Extract feature from position r.
		for(Rectangle r : rect){
			FImage area = image.extractROI(r);
			
			//2D array to 1D array
			float[] vector = ArrayUtils.reshape(area.pixels);
			FloatFV featureV = new FloatFV(vector);
			//Location of rectangle is location of feature
			//SpatialLocation sl = new SpatialLocation(r.x, r.y);
			
			//Generate as a local feature for compatibility with other modules
			//LocalFeature<SpatialLocation, FloatFV> lf = new LocalFeatureImpl<SpatialLocation, FloatFV>(sl,featureV);
			
			areaList.add(vector);
		}
			
		return areaList;	
		 
	}
	
	
	
	public static List<LocalFeature<SpatialLocation, FloatFV>> extract2(FImage image){
        List<LocalFeature<SpatialLocation, FloatFV>> areaList = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

        // Create patch positions
        RectangleSampler rect = new RectangleSampler(image, STEP, STEP, PATCH_SIZE, PATCH_SIZE);

        // Extract feature from position r.
        for(Rectangle r : rect){
            FImage area = image.extractROI(r);

            //2D array to 1D array
            float[] vector = ArrayUtils.reshape(area.pixels);
            FloatFV featureV = new FloatFV(vector);
            //Location of rectangle is location of feature
            SpatialLocation sl = new SpatialLocation(r.x, r.y);
            
            //Generate as a local feature for compatibility with other modules
            LocalFeature<SpatialLocation, FloatFV> lf = new LocalFeatureImpl<SpatialLocation, FloatFV>(sl,featureV);

            areaList.add(lf);
        }

        return areaList;	
	}
	
	
	static class PatchClusterFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
		HardAssigner<float[], float[], IntFloatPair> assigner;

		public PatchClusterFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
			this.assigner = assigner;
		}

        /**
         * Extract features of image, in respect to the HardAssigner.
         * @param image The FImage to use.
         * @return A feature vector.
         */
		public DoubleFV extractFeature(FImage image) {
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
			BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);
			return spatial.aggregate(extract2(image), image.getBounds()).normaliseFV();
			
		}
	}

}
