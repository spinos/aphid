#include <iostream>

#include <HBase.h>
#define MAX_NAME 1024

using namespace aphid;
void
do_dtype(hid_t tid) {

	H5T_class_t t_class;
	t_class = H5Tget_class(tid);
	if(t_class < 0){ 
		puts(" Invalid datatype.\n");
	} else {
		/* 
		 * Each class has specific properties that can be 
		 * retrieved, e.g., size, byte order, exponent, etc. 
		 */
		if(t_class == H5T_INTEGER) {
		      puts(" Datatype is 'H5T_INTEGER'.\n");
			/* display size, signed, endianess, etc. */
		} else if(t_class == H5T_FLOAT) {
		      puts(" Datatype is 'H5T_FLOAT'.\n");
			/* display size, endianess, exponennt, etc. */
		} else if(t_class == H5T_STRING) {
		      puts(" Datatype is 'H5T_STRING'.\n");
			/* display size, padding, termination, etc. */
		} else if(t_class == H5T_BITFIELD) {
		      puts(" Datatype is 'H5T_BITFIELD'.\n");
			/* display size, label, etc. */
		} else if(t_class == H5T_OPAQUE) {
		      puts(" Datatype is 'H5T_OPAQUE'.\n");
			/* display size, etc. */
		} else if(t_class == H5T_COMPOUND) {
		      puts(" Datatype is 'H5T_COMPOUND'.\n");
			/* recursively display each member: field name, type  */
		} else if(t_class == H5T_ARRAY) {
		      puts(" Datatype is 'H5T_COMPOUND'.\n");
			/* display  dimensions, base type  */
		} else if(t_class == H5T_ENUM) {
		      puts(" Datatype is 'H5T_ENUM'.\n");
			/* display elements: name, value   */
		} else  {
		      puts(" Datatype is 'Other'.\n");
		      /* eg. Object Reference, ...and so on ... */
		}
	}
}


void do_attr(hid_t aid) {
	ssize_t len;
	hid_t atype;
	hid_t aspace;
	char buf[MAX_NAME]; 

	/* 
	 * Get the name of the attribute.
	 */
	len = H5Aget_name(aid, MAX_NAME, buf );
	printf("    Attribute Name : %s\n",buf);
	
	/*    
	 * Get attribute information: dataspace, data type 
	 */
	aspace = H5Aget_space(aid); /* the dimensions of the attribute data */
	hsize_t     dims_out[3];
	H5Sget_simple_extent_dims(aspace, dims_out, NULL);
	printf("    Attribute Dimension : %i\n", dims_out[0]);
	
	atype  = H5Aget_type(aid); 
	do_dtype(atype);

	/*
	 * The datatype and dataspace can be used to read all or
	 * part of the data.  (Not shown in this example.)
	 */

	  /* ... read data with H5Aread, write with H5Awrite, etc. */

	H5Tclose(atype);
	H5Sclose(aspace);
}

void
scan_attrs(hid_t oid) {
	int na;
	hid_t aid;
	int i;
	
	na = H5Aget_num_attrs(oid);

	for (i = 0; i < na; i++) {
		aid =	H5Aopen_idx(oid, (unsigned int)i );
		do_attr(aid);
		H5Aclose(aid);
	}
}

void 
do_dset(hid_t did)
{
	hid_t tid;
	hid_t pid;
	hid_t sid;
	hsize_t size;
	char ds_name[MAX_NAME];

        /*
         * Information about the group:
         *  Name and attributes
         *
         *  Other info., not shown here: number of links, object id
         */
	H5Iget_name(did, ds_name, MAX_NAME  );
	printf("Dataset Name : ");
	puts(ds_name);
	printf("\n");

	/*
	 *  process the attributes of the dataset, if any.
	 */
	scan_attrs(did);
  
	/*    
	 * Get dataset information: dataspace, data type 
	 */
	sid = H5Dget_space(did); /* the dimensions of the dataset (not shown) */
	tid = H5Dget_type(did);
	printf(" DATA TYPE:\n");
	do_dtype(tid);

	/*
	 * Retrieve and analyse the dataset properties
	 */
	pid = H5Dget_create_plist(did); /* get creation property list */
	//do_plist(pid);
	size = H5Dget_storage_size(did);
	printf("Total space currently written in file: %d\n",(int)size);
	
	int rank      = H5Sget_simple_extent_ndims(sid);
	
	std::cout<<" n dimension "<<rank<<"\n";
	hsize_t     dims_out[2];
	int status_n  = H5Sget_simple_extent_dims(sid, dims_out, NULL);
    printf(" size %lu x %lu \n",
	   (unsigned long)(dims_out[0]), (unsigned long)(dims_out[1]));
	   
	hid_t plist = H5Dget_create_plist (did);
	
	int numfilt = H5Pget_nfilters (plist);
    printf ("Number of filters associated with dataset: %i\n", numfilt);
	
	for (int i=0; i<numfilt; i++) {
       size_t nelmts = 0;
	   unsigned flags, filter_info;
       H5Z_filter_t filter_type = H5Pget_filter (plist, 0, &flags, &nelmts, NULL, 0, NULL,
                     &filter_info);
       printf ("Filter Type: ");
       switch (filter_type) {
         case H5Z_FILTER_DEFLATE:
              printf ("H5Z_FILTER_DEFLATE\n");
              break;
         case H5Z_FILTER_SZIP:
              printf ("H5Z_FILTER_SZIP\n");
              break;
         default:
              printf ("Other filter type included.\n");
         }
    }

	/*
	float *data = new float[dims_out[0]*dims_out[1]];
	herr_t status = H5Dread(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
                    data);
					
	if(status < 0)
		printf("read data error\n");
		
	for(int i=0; i < dims_out[0]; i++)
		printf("(%f %f %f) ", data[i*3], data[i*3+1], data[i*3+2]);
	//	
	delete[] data;
	*/
	/*
	 * The datatype and dataspace can be used to read all or
	 * part of the data.  (Not shown in this example.)
	 */

	  /* ... read data with H5Dread, write with H5Dwrite, etc. */

	H5Pclose(pid);
	H5Tclose(tid);
	H5Sclose(sid);
}

void 
do_link(hid_t gid, char *name) {
	herr_t status;
	char target[MAX_NAME];

	status = H5Gget_linkval(gid, name, MAX_NAME, target  ) ;
	printf("Symlink: %s points to: %s\n", name, target);
}

void
scan_group(hid_t gid) {
	int i;
	ssize_t len;
	hsize_t nobj;
	herr_t err;
	int otype;
	hid_t grpid, typid, dsid;
	char group_name[MAX_NAME];
	char memb_name[MAX_NAME];

        /*
         * Information about the group:
         *  Name and attributes
         *
         *  Other info., not shown here: number of links, object id
         */
	len = H5Iget_name (gid, group_name, MAX_NAME);

	printf("Group Name: %s\n",group_name);
	
	scan_attrs(gid);
	
	err = H5Gget_num_objs(gid, &nobj);
	
	if(nobj > 0)
		printf("Group %s has %d sub-obj(s)\n",group_name, nobj);
	for (i = 0; i < nobj; i++) {
		// printf("  Member Id: %d ",i);//fflush(stdout);
		len = H5Gget_objname_by_idx(gid, (hsize_t)i, 
			memb_name, (size_t)MAX_NAME );
			
			//printf("   %d ",len);//fflush(stdout);
		// printf("  Member Name: %s ",memb_name);//fflush(stdout);
		otype =  H5Gget_objtype_by_idx(gid, (size_t)i );
		
		switch(otype) {
			case H5G_LINK:
				printf("\n_  SYM_LINK:\n");
				do_link(gid,memb_name);
				break;
			case H5G_GROUP:
				printf("\n_  GROUP:\n");
				grpid = H5Gopen(gid,memb_name, H5P_DEFAULT);
				scan_group(grpid);
				H5Gclose(grpid);
				break;
			case H5G_DATASET:
				printf("\n_  DATASET:\n");
				dsid = H5Dopen(gid,memb_name, H5P_DEFAULT);
				do_dset(dsid);
				H5Dclose(dsid);
				break;
			case H5G_TYPE:
				printf("\n_  DATA TYPE:\n");
				typid = H5Topen(gid,memb_name, H5P_DEFAULT);
				do_dtype(typid);
				H5Tclose(typid);
				break;
			default:
				printf("\n_ unknown?\n");
				break;
			}
	}
}

char diagnoseFile(const char* filename)
{
	hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(file_id < 0) {
		printf("\n cannot open file\n");
		return 0;
	}
	printf("\n begin diagnose file");
	printf(filename);
	hid_t    grp = H5Gopen(file_id,"/", H5P_DEFAULT);
	scan_group(grp);
	H5Gclose(grp);
	herr_t   status = H5Fclose(file_id);
	if(status < 0)
		printf("\n diagnose not closed\n");
	std::cout<<"\n end diagnose\n";
	return 1;
}

int main (int argc, char * const argv[]) {
    if(argc < 2) {
		std::cout<<"\n no input file";
		return 1;
	}
	diagnoseFile(argv[1]);
	return 0;
}
//:~